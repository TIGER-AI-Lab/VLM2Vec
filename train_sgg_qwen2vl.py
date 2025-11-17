import os
import json
import random
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from transformers import HfArgumentParser, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model.model import MMEBModel
from src.model.processor import load_processor,QWEN2_VL,VLM_IMAGE_TOKENS
from src.data.collator.train_collator import MultimodalDataCollator
from typing import Optional
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split


# ------------------------------
# Helper Functions
# ------------------------------
def format_bbox_as_special_token(bbox, normalize=True, original_width=1024, original_height=1024):
    """将边界框转换为Qwen2-VL的special token格式"""
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        
        if normalize:
            x1_norm = int((x1 / original_width) * 1000)
            y1_norm = int((y1 / original_height) * 1000)
            x2_norm = int((x2 / original_width) * 1000)
            y2_norm = int((y2 / original_height) * 1000)
            
            x1_norm = max(0, min(x1_norm, 999))
            y1_norm = max(0, min(y1_norm, 999))
            x2_norm = max(0, min(x2_norm, 999))
            y2_norm = max(0, min(y2_norm, 999))
            
            x1_norm, x2_norm = min(x1_norm, x2_norm), max(x1_norm, x2_norm)
            y1_norm, y2_norm = min(y1_norm, y2_norm), max(y1_norm, y2_norm)
            
            if x1_norm == x2_norm:
                x2_norm = min(x1_norm + 1, 999)
            if y1_norm == y2_norm:
                y2_norm = min(y1_norm + 1, 999)
            
            return f"<|box_start|>({x1_norm}, {y1_norm}), ({x2_norm}, {y2_norm})<|box_end|>"
    return ""

def format_object_with_ref(object_label):
    """将物体标签包装在对象引用token中"""
    return f"<|object_ref_start|>{object_label}<|object_ref_end|>"


# ------------------------------
# Dataset
# ------------------------------
class SGGContrastiveDataset(Dataset):
    def __init__(self, json_path: str, image_dir: str, relation_vocabulary: List[str] = None, num_negatives: int = 5):
        with open(json_path, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                self.samples = json.loads(content)
            else:
                self.samples = [json.loads(l) for l in content.splitlines() if l.strip()]

        self.image_dir = image_dir
        self.num_negatives = num_negatives

        if relation_vocabulary is None:
            rels = set()
            for s in self.samples:
                if 'predicate' in s and s['predicate']:
                    rels.add(s['predicate'])
            self.vocab = sorted(list(rels))
        else:
            self.vocab = relation_vocabulary

        if len(self.vocab) == 0:
            raise ValueError("Empty relation vocabulary")

    def __len__(self):
        return len(self.samples)

    def _full_image_path(self, path: str):
        return path if os.path.isabs(path) else os.path.join(self.image_dir, path)

    def _make_image_field(self, path: str):
        full = self._full_image_path(path)
        return {'resolutions': [None], 'paths': [full], 'bytes': [None]}

    def _get_image_dimensions(self, image_path: str):
        """获取图像尺寸"""
        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            print(f"Warning: Failed to get image dimensions for {image_path}: {e}")
            return (1024, 1024)  # 默认尺寸

    def __getitem__(self, idx):
        s = self.samples[idx]
        image_path = s.get('image_path') or s.get('img_path') or s.get('image')
        predicate = s.get('predicate') or s.get('relation') or "related"
        subj = s.get('subject', {})
        obj = s.get('object', {})
        bbox1 = subj.get('bbox') or s.get('bbox1')
        bbox2 = obj.get('bbox') or s.get('bbox2')
        subj_name = subj.get('class_name', 'objectA')
        obj_name = obj.get('class_name', 'objectB')
        
        # 获取完整图像路径和尺寸
        full_image_path = self._full_image_path(image_path)
        original_width, original_height = self._get_image_dimensions(full_image_path)
        
        # 使用特殊token格式化subject和object
        subj_bbox_token = format_bbox_as_special_token(
            bbox1, True, original_width, original_height
        )
        obj_bbox_token = format_bbox_as_special_token(
            bbox2, True, original_width, original_height
        )
        subj_ref = format_object_with_ref(subj_name)
        obj_ref = format_object_with_ref(obj_name)
        
        # 构建query文本（对齐inference代码）
        query_text = f"{VLM_IMAGE_TOKENS[QWEN2_VL]} In the given image, the subject {subj_ref} is located at {subj_bbox_token}, the object {obj_ref} is located at {obj_bbox_token}. Please return the predicate relationship between the subject and the object."

        # Use full-sentence text for positive and negative targets
        pos_text = f"The subject is {predicate} the object."

         # 生成负样本文本（唯一化）
        neg_candidates = [r for r in self.vocab if r != predicate]
        if neg_candidates:
            negs_raw = random.sample(neg_candidates, min(self.num_negatives, len(neg_candidates)))
        else:
            negs_raw = [predicate] * self.num_negatives
        neg_texts = [f"The subject is {r} the object." for r in negs_raw]

        # 生成 image field（只生成一次，复用）
        img_field = self._make_image_field(image_path)
        query_image = img_field
        pos_image = img_field
        neg_images = [img_field] * len(neg_texts)

        return {
            'query_text': query_text,
            'query_image': query_image,
            'pos_text': pos_text,
            'pos_image': pos_image,
            'neg_text': neg_texts,
            'neg_image': neg_images,
            'global_dataset_name': s.get('dataset_name', 'vg')
        }


# ------------------------------
# Utils
# ------------------------------
def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(x, device) for x in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch


def plot_loss_curve(loss_history: Dict[str, List[float]], save_path: str):
    """绘制训练 loss 曲线"""
    plt.figure(figsize=(14, 6))
    
    # Plot step-wise loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history['steps'], loss_history['losses'], alpha=0.6, label='Train Loss')
    if len(loss_history['losses']) > 100:
        # 平滑曲线
        window = min(100, len(loss_history['losses']) // 10)
        smoothed = np.convolve(loss_history['losses'], np.ones(window)/window, mode='valid')
        plt.plot(range(window//2, len(smoothed) + window//2), smoothed, 
                linewidth=2, label=f'Smoothed (window={window})')
    # Plot eval losses recorded at eval steps (if any)
    if 'eval_steps' in loss_history and loss_history.get('eval_steps'):
        plt.scatter(loss_history['eval_steps'], loss_history.get('eval_losses_steps', []),
                    color='red', marker='x', s=40, label='Eval Loss (step)')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss (Step-wise)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot epoch-wise average loss
    plt.subplot(1, 2, 2)
    if loss_history['epoch_losses']:
        epochs = range(1, len(loss_history['epoch_losses']) + 1)
        plt.plot(epochs, loss_history['epoch_losses'], marker='o', linewidth=2, markersize=8, label='Train Loss')
        
        # Plot eval loss if available
        if 'eval_losses' in loss_history and loss_history['eval_losses']:
            eval_epochs = range(1, len(loss_history['eval_losses']) + 1)
            plt.plot(eval_epochs, loss_history['eval_losses'], marker='s', linewidth=2, markersize=8, label='Eval Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Loss (Epoch-wise)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 标注最小 loss
        min_loss_epoch = np.argmin(loss_history['epoch_losses']) + 1
        min_loss = loss_history['epoch_losses'][min_loss_epoch - 1]
        plt.annotate(f'Min: {min_loss:.4f}', 
                    xy=(min_loss_epoch, min_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Loss curve saved to: {save_path}")


def evaluate(model, val_loader, device, return_scores=False):
    """
    评估模型平均 InfoNCE loss，同时可选择返回最后一个 batch 的 scores。
    """
    model.eval()
    losses = []
    last_scores = None

    with torch.no_grad():
        for batch in val_loader:
            qry_inputs, pos_inputs, neg_inputs = batch  # collator 返回三部分

            qry_inputs = batch_to_device(qry_inputs, device)
            pos_inputs = batch_to_device(pos_inputs, device)
            neg_inputs = batch_to_device(neg_inputs, device)

            # forward 返回 dict 包含 loss 和 scores
            out = model(qry=qry_inputs, tgt=pos_inputs, neg=neg_inputs)
            loss_tensor = out["loss"]
            scores = out.get("scores", None)

            losses.append(loss_tensor.item())
            last_scores = scores  # 记录最后一个 batch 的 scores

    model.train()
    mean_loss = float(np.mean(losses)) if losses else 0.0
    if return_scores:
        return mean_loss, last_scores
    return mean_loss



def save_loss_json(loss_history: Dict[str, List[float]], save_path: str):
    """保存 loss 数据为 JSON"""
    with open(save_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"💾 Loss data saved to: {save_path}")


# ------------------------------
# Train Loop
# ------------------------------
def train_loop(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading processor...")
    processor = load_processor(model_args)

    print("Building model...")
    model = MMEBModel.build(model_args)
    model = model.to(device)
    model.train()

    # 应用 LoRA
    if model_args.lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target_modules.split(","),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )
        model.encoder = get_peft_model(model.encoder, lora_config)
        print("✅ LoRA applied")
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # 准备数据
    dataset = SGGContrastiveDataset(
        data_args.dataset_json, 
        data_args.image_dir, 
        num_negatives=data_args.num_negatives
    )
    collator = MultimodalDataCollator(
        processor=processor,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        batch_size=training_args.per_device_train_batch_size
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=training_args.per_device_train_batch_size, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=0,  # 可根据需要调整
        pin_memory=True if torch.cuda.is_available() else False
    )

    # ---------- evaluation dataset/loader (optional) ----------
    # Use the eval path supplied in DataArguments. The default should be set in `src/arguments.py`.
    eval_json = getattr(data_args, 'eval_dataset_json', None)
    if eval_json:
        try:
            val_dataset = SGGContrastiveDataset(
                eval_json,
                data_args.image_dir,
                num_negatives=data_args.num_negatives
            )
            eval_batch_size = getattr(training_args, 'per_device_eval_batch_size', training_args.per_device_train_batch_size)
            val_loader = DataLoader(
                val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False
            )
            print(f"✅ Eval dataset loaded: {len(val_dataset)} samples from {eval_json}")
        except Exception as e:
            val_loader = None
            print(f"⚠️  Failed to load eval dataset from {eval_json}: {e}")
    else:
        val_loader = None
        print("ℹ️  No eval_dataset_json provided in DataArguments; skipping evaluation during training.")

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay if hasattr(training_args, 'weight_decay') else 0.01
    )

    # 计算总步数
    num_update_steps_per_epoch = len(dataloader) // training_args.gradient_accumulation_steps
    total_steps = int(training_args.num_train_epochs) * num_update_steps_per_epoch
    warmup_steps = int(total_steps * (training_args.warmup_ratio if hasattr(training_args, 'warmup_ratio') else 0.1))

    # Cosine 学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"\n🚀 Training Info:")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Num epochs: {training_args.num_train_epochs}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print()

    # Loss 记录
    loss_history = {
        'steps': [],
        'losses': [],
        'epoch_losses': [],
        'eval_losses': [],  # eval loss at each epoch
        'eval_losses_steps': [],  # eval loss at each eval step
        'eval_steps': []  # step indices for eval loss
    }

    global_step = 0
    best_loss = float('inf')
    best_eval_loss = float('inf')

    # 训练循环
    for epoch in range(int(training_args.num_train_epochs)):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{int(training_args.num_train_epochs)}")
        print(f"{'='*60}")
        
        epoch_losses = []
        optimizer.zero_grad()
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        
        for batch_idx, (qry_inputs, pos_inputs, neg_inputs) in enumerate(dataloader):
            qry_inputs = batch_to_device(qry_inputs, device)
            pos_inputs = batch_to_device(pos_inputs, device)
            neg_inputs = batch_to_device(neg_inputs, device)
            # 前向传播
            loss = model(qry=qry_inputs, tgt=pos_inputs, neg=neg_inputs)
            loss_tensor = loss["loss"] if isinstance(loss, dict) else loss
            
            # 梯度累积
            loss_tensor = loss_tensor / training_args.gradient_accumulation_steps
            loss_tensor.backward()

            # 记录 loss
            epoch_losses.append(loss_tensor.item() * training_args.gradient_accumulation_steps)

            # 更新参数
            if (batch_idx + 1) % training_args.gradient_accumulation_steps == 0:
                # 梯度裁剪(可选)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 记录到 loss history
                loss_history['steps'].append(global_step)
                loss_history['losses'].append(loss_tensor.item() * training_args.gradient_accumulation_steps)

                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss_tensor.item() * training_args.gradient_accumulation_steps:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

                # 定期打印
                if global_step % training_args.logging_steps == 0:
                    avg_loss = np.mean(epoch_losses[-training_args.logging_steps:])
                    print(f"\n[Step {global_step}] loss: {loss_tensor.item() * training_args.gradient_accumulation_steps:.6f} | avg_loss: {avg_loss:.6f} | lr: {scheduler.get_last_lr()[0]:.2e}")

                global_step += 1

                # 定期 evaluation（如果配置了 eval_steps 并且成功加载了 val_loader）
                if hasattr(training_args, 'eval_steps') and getattr(training_args, 'eval_steps') and val_loader is not None:
                    if global_step % int(training_args.eval_steps) == 0:
                        val_loss = evaluate(model, val_loader, device)
                        print(f"\n[Eval at step {global_step}] val_loss: {val_loss:.6f}")
                        # Track eval loss at this step
                        loss_history['eval_losses_steps'].append(val_loss)
                        loss_history['eval_steps'].append(global_step)

                        # 根据 eval loss 保存最佳模型（可选）
                        if val_loss < best_eval_loss:
                            best_eval_loss = val_loss
                            best_dir_eval = os.path.join(training_args.output_dir, "best_model_eval")
                            os.makedirs(best_dir_eval, exist_ok=True)
                            model.save(best_dir_eval)
                            print(f"🏆 New best eval model saved (val_loss: {best_eval_loss:.6f}) to: {best_dir_eval}")

        # Epoch 结束
        avg_epoch_loss = np.mean(epoch_losses)
        loss_history['epoch_losses'].append(avg_epoch_loss)

        # Evaluate at the end of each epoch and track eval loss
        if val_loader is not None:
            val_loss_epoch = evaluate(model, val_loader, device)
            loss_history['eval_losses'].append(val_loss_epoch)
            print(f"\n[Eval at epoch {epoch+1}] val_loss: {val_loss_epoch:.6f}")
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.6f}")
        print(f"  Min Loss: {min(epoch_losses):.6f}")
        print(f"  Max Loss: {max(epoch_losses):.6f}")

        # 保存 checkpoint
        if (epoch + 1) % training_args.save_steps == 0 or (epoch + 1) == int(training_args.num_train_epochs):
            save_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model.save(save_dir)
            print(f"💾 Checkpoint saved to: {save_dir}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_dir = os.path.join(training_args.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save(best_dir)
            print(f"🏆 Best model saved (loss: {best_loss:.6f}) to: {best_dir}")

    # 最终保存
    final_dir = os.path.join(training_args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save(final_dir)
    
    # 最终的 loss 图和数据
    plot_loss_curve(loss_history, os.path.join(training_args.output_dir, "final_loss_curve.png"))
    save_loss_json(loss_history, os.path.join(training_args.output_dir, "final_loss_history.json"))
    
    print(f"\n{'='*60}")
    print("✅ Training Complete!")
    print(f"{'='*60}")
    print(f"📁 Final model saved to: {final_dir}")
    print(f"🏆 Best model saved to: {os.path.join(training_args.output_dir, 'best_model')}")
    print(f"📊 Loss curve: {os.path.join(training_args.output_dir, 'final_loss_curve.png')}")
    print(f"💾 Loss data: {os.path.join(training_args.output_dir, 'final_loss_history.json')}")
    print(f"🎯 Best loss: {best_loss:.6f}")


# ------------------------------
# Main
# ------------------------------
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    # 使用 checkpoint path 作为 model name (离线加载)
    if model_args.checkpoint_path is not None:
        model_args.model_name = model_args.checkpoint_path

    train_loop(model_args, data_args, training_args)


if __name__ == "__main__":
    main()