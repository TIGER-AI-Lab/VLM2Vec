import os
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import get_scheduler
from torch.cuda.amp import autocast, GradScaler

from src.model.model import MMEBModel
from src.arguments import ModelArguments, DataArguments
from src.model.processor import load_processor, VLM_IMAGE_TOKENS, QWEN2_VL

# -------------------------
# 配置参数
# -------------------------
MODEL_NAME = '/public/home/wangby2025/plusLab/VLM2Vec/model/Qwen/Qwen2-VL-2B-Instruct'
CHECKPOINT_PATH = '/public/home/wangby2025/plusLab/VLM2Vec/model/TIGER-Lab/VLM2Vec-Qwen2VL-2B'
INPUT_FILE = '/public/home/wangby2025/plusLab/data/vg/train_1200.json'
DEVICE = 'cuda'
BATCH_SIZE = 1   # 降到1避免显存爆掉
EPOCHS = 15
LR = 3e-4
LORA_RANK = 4
LORA_ALPHA = 32
TEMPERATURE = 0.07
MAX_TOKEN_LENGTH = 128
BEST_MODEL_DIR = "./loraModel/best_vlm2vec_lora"
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

PREDICATES = [
    "above", "across", "against", "along", "and", "at", "attached to", "behind",
    "belonging to", "between", "carrying", "covered in", "covering", "eating",
    "flying in", "for", "from", "growing on", "hanging from", "has", "holding",
    "in", "in front of", "laying on", "looking at", "lying on", "made of",
    "mounted on", "near", "of", "on", "on back of", "over", "painted on",
    "parked on", "part of", "playing", "riding", "says", "sitting on",
    "standing on", "to", "under", "using", "walking in", "walking on",
    "watching", "wearing", "wears", "with"
]

# -------------------------
# 数据集
# -------------------------
class VGPredicateDataset(Dataset):
    def __init__(self, data_json, processor):
        with open(data_json, 'r') as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        subject = item['subject']
        object_ = item['object']
        predicate = item['predicate']

        image = Image.open(image_path).convert('RGB')

        subj_bbox = f"<|box_start|>({subject['bbox'][0]}, {subject['bbox'][1]}), ({subject['bbox'][2]}, {subject['bbox'][3]})<|box_end|>"
        obj_bbox = f"<|box_start|>({object_['bbox'][0]}, {object_['bbox'][1]}), ({object_['bbox'][2]}, {object_['bbox'][3]})<|box_end|>"

        subj_ref = f"<|object_ref_start|>{subject['class_name']}<|object_ref_end|>"
        obj_ref = f"<|object_ref_start|>{object_['class_name']}<|object_ref_end|>"

        query_text = f"{VLM_IMAGE_TOKENS[QWEN2_VL]} In the given image, the subject {subj_ref} is located at {subj_bbox}, the object {obj_ref} is located at {obj_bbox}. Return the predicate relationship."

        inputs = self.processor(
            text=query_text,
            images=image,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_TOKEN_LENGTH
        )

        return inputs, predicate

# -------------------------
# collate_fn
# -------------------------
def collate_fn(batch):
    inputs_list, predicates = zip(*batch)
    batch_inputs = {}

    # 文本 padding
    for k in ['input_ids', 'attention_mask']:
        batch_inputs[k] = torch.nn.utils.rnn.pad_sequence(
            [x[k].squeeze(0) for x in inputs_list],
            batch_first=True,
            padding_value=0
        )

    # 图像处理
    processed_pv = []
    max_H, max_W = 0, 0

    for x in inputs_list:
        pv = x['pixel_values']
        if pv.dim() == 2:
            pv = pv.unsqueeze(0).unsqueeze(0)
        elif pv.dim() == 3:
            pv = pv.unsqueeze(0)
        N, C, H, W = pv.shape
        max_H = max(max_H, H)
        max_W = max(max_W, W)
        processed_pv.append(pv)

    padded_pv = []
    for pv in processed_pv:
        N, C, H, W = pv.shape
        pad_H = max_H - H
        pad_W = max_W - W
        pv = torch.nn.functional.pad(pv, (0, pad_W, 0, pad_H))
        if C == 1:
            pv = pv.repeat(1, 3, 1, 1)
        elif C > 3:
            pv = pv[:, :3, :, :]
        pv = pv.float() / 255.0
        padded_pv.append(pv)

    batch_inputs['pixel_values'] = torch.cat(padded_pv, dim=0)

    return batch_inputs, list(predicates)

# -------------------------
# InfoNCE Loss
# -------------------------
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeds, predicate_embeds, target_idx):
        sim_matrix = torch.matmul(query_embeds, predicate_embeds.transpose(1,2)) / self.temperature
        sim_matrix = sim_matrix.squeeze(1)
        loss = nn.functional.cross_entropy(sim_matrix, target_idx)
        return loss

# -------------------------
# 初始化模型 & LoRA
# -------------------------
model_args = ModelArguments(
    model_name=MODEL_NAME,
    checkpoint_path=CHECKPOINT_PATH,
    pooling='last',
    normalize=True,
    model_backbone='qwen2_vl',
    lora=True
)
data_args = DataArguments(
    resize_min_pixels=56*56,
    resize_max_pixels=28*28*1280
)

processor = load_processor(model_args, data_args)
model = MMEBModel.load(model_args).to(DEVICE)
model.train()

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

base_model = model.encoder
base_model = get_peft_model(base_model, lora_config)
model.encoder = base_model

# -------------------------
# 生成 predicate embedding
# -------------------------
def get_predicate_embeddings(predicates, processor, model):
    pred_inputs = processor(
        text=predicates,
        images=None,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MAX_TOKEN_LENGTH
    )
    device = next(model.parameters()).device
    pred_inputs = {k: v.to(device) for k, v in pred_inputs.items()}

    with torch.no_grad():
        outputs = model(tgt=pred_inputs)
        pred_embeds = outputs["tgt_reps"]

    return pred_embeds.detach().cpu()

pred_embeds_all = get_predicate_embeddings(PREDICATES, processor, model)

# -------------------------
# DataLoader
# -------------------------
dataset = VGPredicateDataset(INPUT_FILE, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# -------------------------
# Optimizer & Scheduler
# -------------------------
optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = len(dataloader) * EPOCHS
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
criterion = InfoNCELoss(temperature=TEMPERATURE)
scaler = GradScaler()

# -------------------------
# 训练循环
# -------------------------
best_loss = float('inf')
best_epoch = -1
epoch_losses = []

for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    pbar = tqdm(dataloader)
    epoch_loss = 0.0

    for batch_inputs, batch_predicates in pbar:
        batch_inputs = {k: v.to(DEVICE) for k, v in batch_inputs.items()}
        batch_size = batch_inputs['input_ids'].size(0)

        query_embeds_list = []
        target_idx_list = []

        for i in range(batch_size):
            single_input = {
                'input_ids': batch_inputs['input_ids'][i].unsqueeze(0),
                'attention_mask': batch_inputs['attention_mask'][i].unsqueeze(0),
                'pixel_values': batch_inputs['pixel_values'][i].unsqueeze(0)
            }
            target_idx = PREDICATES.index(batch_predicates[i])
            target_idx_list.append(target_idx)

            with autocast():
                qry_emb = model(qry=single_input)["qry_reps"].squeeze(0)
            query_embeds_list.append(qry_emb)

        query_embeds = torch.stack(query_embeds_list, dim=0)
        predicate_embeds = pred_embeds_all.unsqueeze(0).to(DEVICE)
        target_idx = torch.tensor(target_idx_list, device=DEVICE)

        with autocast():
            loss = criterion(query_embeds=query_embeds, predicate_embeds=predicate_embeds, target_idx=target_idx)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()
        pbar.set_description(f"Loss: {loss.item():.4f}")

        torch.cuda.empty_cache()

    avg_epoch_loss = epoch_loss / len(dataloader)
    epoch_losses.append(avg_epoch_loss)
    print(f"📉 Epoch {epoch+1} Average Loss: {avg_epoch_loss:.6f}")

    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        best_epoch = epoch + 1
        model.save_pretrained(BEST_MODEL_DIR)
        print(f"✅ New best model saved at epoch {best_epoch} with loss {best_loss:.6f}")

# -------------------------
# 绘制 Loss 曲线
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(range(1, EPOCHS+1), epoch_losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("./loraModel/loss_curve.png")
plt.close()
print("📈 Loss curve saved to ./loraModel/loss_curve.png")

print("\n🎉 训练完成！")
print(f"🏆 最佳模型来自第 {best_epoch} 轮，平均Loss={best_loss:.6f}")
print(f"📂 权重保存在: {BEST_MODEL_DIR}")
