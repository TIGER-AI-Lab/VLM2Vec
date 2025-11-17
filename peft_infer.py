import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor

# 导入训练时使用的模型类和参数
from src.model.model import MMEBModel
from src.arguments import ModelArguments

# ---------------- 输入输出 ----------------
INPUT_FILE = "/public/home/wangby2025/plusLab/data/vg/test.json"
OUTPUT_FILE = "/public/home/wangby2025/plusLab/outputs/sgg_qwen2vl/recall_results.json"

# ---------------- 谓词列表 ----------------
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

# ---------------- 辅助函数 ----------------
def format_bbox_as_special_token(bbox, normalize=True, original_width=1024, original_height=1024):
    x1, y1, x2, y2 = bbox
    if normalize:
        x1 = max(0, min(int((x1 / original_width) * 1000), 999))
        y1 = max(0, min(int((y1 / original_height) * 1000), 999))
        x2 = max(0, min(int((x2 / original_width) * 1000), 999))
        y2 = max(0, min(int((y2 / original_height) * 1000), 999))
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x1 == x2: x2 = min(x1 + 1, 999)
        if y1 == y2: y2 = min(y1 + 1, 999)
    return f"<|box_start|>({x1}, {y1}), ({x2}, {y2})<|box_end|>"

def format_object_with_ref(label):
    return f"<|object_ref_start|>{label}<|object_ref_end|>"

def cosine_similarity(a, b):
    return (a @ b.T) / (a.norm() * b.norm() + 1e-8)

# ---------------- 预测函数 ----------------
def predict_relation(model, processor, image_path, subj_obj, obj_obj, width, height, device="cuda"):
    """
    预测主语和宾语之间的关系
    """
    # 读取图像
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"⚠️ 无法读取图像 {image_path}: {e}")
        return None
    
    # 构建带位置信息的 token
    subj_token = format_object_with_ref(subj_obj['class_name']) + format_bbox_as_special_token(
        subj_obj['bbox'], True, width, height
    )
    obj_token = format_object_with_ref(obj_obj['class_name']) + format_bbox_as_special_token(
        obj_obj['bbox'], True, width, height
    )
    
    query_text = f"In the image, subject {subj_token}, object {obj_token}. What's their relationship?"
    
    # 方法1：使用 Qwen2-VL 的标准 chat template 格式
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query_text}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        query_inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(device)
        
    except Exception as e:
        # 方法2：如果上面的方法失败，尝试直接处理
        print(f"⚠️ 使用标准格式失败，尝试备用方案: {e}")
        try:
            query_inputs = processor(
                text=query_text,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)
        except Exception as e2:
            print(f"❌ 备用方案也失败: {e2}")
            return None
    
    # 使用模型编码查询
    with torch.no_grad():
        query_emb = model.encode_input(query_inputs)
    
    # 对每个谓词计算相似度
    predicate_scores = []
    for pred in PREDICATES:
        # 编码谓词（纯文本，不需要图像）
        tgt_inputs = processor(text=pred, return_tensors="pt").to(device)
        
        with torch.no_grad():
            pred_emb = model.encode_input(tgt_inputs)
        
        # 计算余弦相似度
        if query_emb.dim() == 2 and pred_emb.dim() == 2:
            sim = cosine_similarity(query_emb, pred_emb)
            sim_score = sim.item()
        else:
            # 处理 1D tensor
            q_emb = query_emb.unsqueeze(0) if query_emb.dim() == 1 else query_emb
            p_emb = pred_emb.unsqueeze(0) if pred_emb.dim() == 1 else pred_emb
            sim = cosine_similarity(q_emb, p_emb)
            sim_score = sim.item()
        
        predicate_scores.append({'predicate': pred, 'similarity': sim_score})
    
    return predicate_scores

# ---------------- Recall计算 ----------------
def calculate_recall_per_image(candidates, k=50):
    """计算单张图像的 Recall@k"""
    sorted_preds = sorted(candidates, key=lambda x: x['similarity'], reverse=True)[:k]
    recalled = set([c['relation_idx'] for c in sorted_preds if c['is_correct']])
    total_gt = len(set(c['relation_idx'] for c in candidates))
    recall = len(recalled) / total_gt if total_gt > 0 else 0.0
    return recall, len(recalled), total_gt

# ---------------- 主函数 ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    backbone_path = '/public/home/wangby2025/plusLab/VLM2Vec/model/Qwen/Qwen2-VL-2B-Instruct'
    lora_path = '/public/home/wangby2025/plusLab/outputs/sgg_qwen2vl/final'
    
    print("=" * 60)
    print("🚀 开始加载模型...")
    print("=" * 60)
    
    # 创建 ModelArguments
    from dataclasses import fields
    valid_fields = {f.name for f in fields(ModelArguments)}
    
    args_dict = {
        'model_name': backbone_path,
        'checkpoint_path': lora_path,
    }
    
    optional_params = {
        'lora': True,
        'normalize': True,
        'temperature': 0.02,
        'pooling_method': 'last',
        'pooling': 'last',
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'lora_target_modules': 'q_proj,k_proj,v_proj,o_proj',
    }
    
    for key, value in optional_params.items():
        if key in valid_fields:
            args_dict[key] = value
    
    filtered_args = {k: v for k, v in args_dict.items() if k in valid_fields}
    model_args = ModelArguments(**filtered_args)
    
    # 加载模型
    try:
        print("尝试使用 MMEBModel.load...")
        model = MMEBModel.load(model_args, is_trainable=False)
        print("✅ 使用 MMEBModel.load 加载成功")
    except Exception as e:
        print(f"MMEBModel.load 失败: {e}")
        print("尝试使用 MMEBModel.build...")
        
        model = MMEBModel.build(model_args)
        
        if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            try:
                from peft import PeftModel
                print("加载 LoRA adapter...")
                model.encoder = PeftModel.from_pretrained(
                    model.encoder, 
                    lora_path,
                    is_trainable=False
                )
                model.encoder = model.encoder.merge_and_unload()
                print("✅ LoRA 权重加载并合并成功")
            except Exception as e:
                print(f"⚠️ LoRA 加载失败: {e}")
    
    model = model.to(device)
    model.eval()
    print(f"✅ 模型已加载到设备: {next(model.parameters()).device}")

    # 🔥 关键修改：使用 AutoProcessor 而不是 AutoTokenizer
    print("\n加载 Processor...")
    processor = AutoProcessor.from_pretrained(
        backbone_path, 
        trust_remote_code=True
    )
    print("✅ Processor 加载成功")

    # 读取测试数据
    print("\n" + "=" * 60)
    print("📖 读取测试数据...")
    print("=" * 60)
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"✅ 总共 {len(data)} 条关系数据")

    # 按图片分组
    relations_by_image = {}
    for item in data:
        img_id = item.get('image_id')
        if img_id not in relations_by_image:
            relations_by_image[img_id] = []
        relations_by_image[img_id].append(item)

    print(f"✅ 涉及 {len(relations_by_image)} 张图片")

    # 初始化结果存储
    per_image_candidates = {}
    all_relations_info = []
    global_idx = 0
    total_candidates = 0
    failed_count = 0

    # 批量预测
    print("\n" + "=" * 60)
    print("🔮 开始预测关系...")
    print("=" * 60)
    
    for img_id, relations in tqdm(relations_by_image.items(), desc="处理图片"):
        # 从第一条关系获取图片路径
        img_path = relations[0]['image_path']
        
        try:
            width, height = Image.open(img_path).size
        except Exception as e:
            print(f"\n⚠️ 无法打开图片 {img_path}: {e}")
            failed_count += len(relations)
            continue
        
        image_candidates = []
        image_relation_idx = 0

        for rel in relations:
            subj_obj = rel['subject']
            obj_obj = rel['object']
            gt_predicate = rel['predicate']

            # 🔥 关键修改：传入图像路径和 processor
            try:
                pred_scores = predict_relation(
                    model, processor, img_path, 
                    subj_obj, obj_obj, width, height, device
                )
                
                if pred_scores is None:
                    failed_count += 1
                    continue
                    
            except Exception as e:
                print(f"\n⚠️ 预测失败 (image {img_id}, relation {image_relation_idx}): {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue

            # 记录关系信息
            all_relations_info.append({
                'relation_idx': global_idx,
                'image_id': img_id,
                'image_relation_idx': image_relation_idx,
                'subject': subj_obj['class_name'],
                'object': obj_obj['class_name'],
                'gt_predicate': gt_predicate
            })

            # 保存候选
            for p in pred_scores:
                image_candidates.append({
                    'relation_idx': image_relation_idx,
                    'global_relation_idx': global_idx,
                    'image_id': img_id,
                    'subject': subj_obj['class_name'],
                    'object': obj_obj['class_name'],
                    'gt_predicate': gt_predicate,
                    'predicted_predicate': p['predicate'],
                    'similarity': p['similarity'],
                    'is_correct': p['predicate'] == gt_predicate
                })
                total_candidates += 1

            image_relation_idx += 1
            global_idx += 1

        per_image_candidates[img_id] = image_candidates

    if failed_count > 0:
        print(f"\n⚠️ 共有 {failed_count} 条关系预测失败")

    # 计算 Per-Image Recall@50
    print("\n" + "=" * 60)
    print("📊 计算 Recall 指标...")
    print("=" * 60)
    
    per_image_results = []
    for img_id, candidates in per_image_candidates.items():
        if len(candidates) == 0:
            continue
        recall, recalled, total_gt = calculate_recall_per_image(candidates, k=50)
        per_image_results.append({
            'image_id': img_id,
            'recall@50': recall,
            'recalled_relations': recalled,
            'total_gt_relations': total_gt
        })

    avg_recall = sum(r['recall@50'] for r in per_image_results) / len(per_image_results) if per_image_results else 0
    total_recalled_relations = sum(r['recalled_relations'] for r in per_image_results)
    total_gt_relations = sum(r['total_gt_relations'] for r in per_image_results)

    # 全局 top50 候选
    all_candidate_predictions = []
    for candidates in per_image_candidates.values():
        all_candidate_predictions.extend(candidates)
    top50_global_candidates = sorted(all_candidate_predictions, key=lambda x: x['similarity'], reverse=True)[:50]

    # 构建输出
    output_data = {
        'summary': {
            'evaluation_method': 'per-image',
            'total_images': len(per_image_candidates),
            'total_relations': len(all_relations_info),
            'total_candidates': total_candidates,
            'failed_predictions': failed_count,
            'avg_recall@50': avg_recall,
            'total_recalled_relations': total_recalled_relations,
            'total_gt_relations': total_gt_relations
        },
        'per_image_results': per_image_results,
        'all_relations': all_relations_info,
        'top50_global_candidates': top50_global_candidates
    }

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("✅ 评估完成！")
    print("=" * 60)
    print(f"📈 平均 Recall@50: {avg_recall:.4f}")
    print(f"📊 总召回关系数: {total_recalled_relations}/{total_gt_relations}")
    print(f"💾 结果已保存至: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()