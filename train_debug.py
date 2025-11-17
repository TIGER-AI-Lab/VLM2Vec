import torch
from torch.utils.data import DataLoader
from src.model.model import MMEBModel
from src.model.processor import load_processor
from src.data.collator.train_collator import MultimodalDataCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from train_sgg_qwen2vl import SGGContrastiveDataset, batch_to_device
import random

# -----------------------------
# 配置（替换成你的路径）
# -----------------------------
model_args = ModelArguments(
    model_name="/public/home/wangby2025/plusLab/VLM2Vec/model/Qwen/Qwen2-VL-2B-Instruct",
    model_backbone="qwen2_vl",
    checkpoint_path="/public/home/wangby2025/plusLab/VLM2Vec/model/Qwen/Qwen2-VL-2B-Instruct",
    lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
)

data_args = DataArguments(
    dataset_json="/public/home/wangby2025/plusLab/data/vg/train_100.json",
    image_dir="/public/home/wangby2025/plusLab/data/vg/VG150",
    num_negatives=4
)

training_args = TrainingArguments(
    per_device_train_batch_size=2,  # 小 batch
    gradient_accumulation_steps=1,
    fp16=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 加载 processor
# -----------------------------
processor = load_processor(model_args)

# -----------------------------
# 构建模型
# -----------------------------
model = MMEBModel.build(model_args)
model = model.to(device)
model.train()

# -----------------------------
# 应用 LoRA
# -----------------------------
if model_args.lora:
    from peft import LoraConfig, get_peft_model
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

# -----------------------------
# 构建 Dataset 和 DataLoader
# -----------------------------
dataset = SGGContrastiveDataset(
    data_args.dataset_json, 
    data_args.image_dir, 
    num_negatives=data_args.num_negatives
)

# 随机抽取少量样本做调试
sample_indices = random.sample(range(len(dataset)), min(8, len(dataset)))
sample_dataset = torch.utils.data.Subset(dataset, sample_indices)

collator = MultimodalDataCollator(
    processor=processor,
    model_args=model_args,
    data_args=data_args,
    training_args=training_args,
    batch_size=training_args.per_device_train_batch_size
)

dataloader = DataLoader(
    sample_dataset,
    batch_size=training_args.per_device_train_batch_size,
    collate_fn=collator,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

# -----------------------------
# 前向计算测试
# -----------------------------
for batch_idx, (qry_inputs, pos_inputs) in enumerate(dataloader):
    qry_inputs = batch_to_device(qry_inputs, device)
    pos_inputs = batch_to_device(pos_inputs, device)
    
    loss = model(qry=qry_inputs, tgt=pos_inputs)
    loss_value = loss["loss"].item() if isinstance(loss, dict) else loss.item()
    
    print(f"[DEBUG] Batch {batch_idx} loss: {loss_value:.6f}")
    
    # 打印部分 query/positive/negative 内容
    print("Sample query text:", qry_inputs.get("input_ids")[0][:20] if "input_ids" in qry_inputs else "N/A")
    print("Sample pos text:", pos_inputs.get("input_ids")[0][:20] if "input_ids" in pos_inputs else "N/A")
    
    break  # 只看第一个 batch
