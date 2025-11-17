      

import json
import torch
from PIL import Image
from tqdm import tqdm
import os
import sys
import warnings


def check_flash_attention_support():

    try:
        # 检查是否有可用的GPU
        if not torch.cuda.is_available():
            return False, "CUDA不可用"
        
        # 获取GPU计算能力
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability
        compute_capability = major * 10 + minor
        
        # Flash Attention 2需要计算能力 >= 8.0 (Ampere及以上架构)
        # Flash Attention 1需要计算能力 >= 7.5 (Turing及以上架构)
        if compute_capability >= 80:
            # 尝试导入flash_attn
            try:
                import flash_attn
                return True, f"支持Flash Attention (GPU计算能力: {major}.{minor})"
            except ImportError:
                return False, f"GPU支持但未安装flash_attn包 (计算能力: {major}.{minor})"
        else:
            return False, f"GPU计算能力不足 (当前: {major}.{minor}, 需要: >= 8.0)"
            
    except Exception as e:
        return False, f"检测失败: {str(e)}"


def configure_attention_backend():

    is_supported, message = check_flash_attention_support()
    
    print("\n" + "="*80)
    print("注意力机制配置")
    print("="*80)
    
    if is_supported:
        print(f"✅ {message}")
        print("   使用: Flash Attention (最快)")
        os.environ["ATTN_IMPLEMENTATION"] = "flash_attention_2"
        # 同时设置transformers使用的环境变量
        os.environ["USE_FLASH_ATTENTION"] = "1"
        return "flash_attn"
    else:
        print(f"⚠️  {message}")
        
        # 检查PyTorch版本是否支持SDPA
        pytorch_version = torch.__version__
        major, minor = map(int, pytorch_version.split('.')[:2])
        
        if major >= 2:  # PyTorch 2.0+支持SDPA
            print("   降级使用: Scaled Dot Product Attention (SDPA)")
            print("   性能: 中等，但比eager模式快")
            os.environ["ATTN_IMPLEMENTATION"] = "sdpa"
            os.environ["USE_FLASH_ATTENTION"] = "0"
            return "sdpa"
        else:
            print("   降级使用: Eager Attention (标准实现)")
            print("   性能: 较慢，但兼容性最好")
            os.environ["ATTN_IMPLEMENTATION"] = "eager"
            os.environ["USE_FLASH_ATTENTION"] = "0"
            return "eager"
    
    print("="*80 + "\n")


_attn_type = configure_attention_backend()

# 现在才导入VLM2Vec模块
from src.model.model import MMEBModel
from src.arguments import ModelArguments, DataArguments
from src.model.processor import load_processor, QWEN2_VL, VLM_IMAGE_TOKENS


INPUT_FILE = "/public/home/xiaojw2025/Workspace/RAHP/DATASET/VG150/test_200_images.json"
OUTPUT_FILE = "/public/home/xiaojw2025/Workspace/VLM2Vec/predict/recall_results_200.json"

# 50个谓词列表
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
    return f"<|object_ref_start|>{object_label}<|object_ref_end|>"


def predict_relation(model, processor, image_path, subject_obj, object_obj, original_width, original_height):
    # 构建subject和object的特殊token
    subj_bbox_token = format_bbox_as_special_token(
        subject_obj['bbox'], True, original_width, original_height
    )
    obj_bbox_token = format_bbox_as_special_token(
        object_obj['bbox'], True, original_width, original_height
    )
    subj_ref = format_object_with_ref(subject_obj['class_name'])
    obj_ref = format_object_with_ref(object_obj['class_name'])
    
    # 构建query文本（图像+物体位置信息）
    query_text = f"{VLM_IMAGE_TOKENS[QWEN2_VL]} In the given image, the subject {subj_ref} is located at {subj_bbox_token},the object{obj_ref} is located at {obj_bbox_token}.Please return the predicate relationship between the subject and the object."
    
    # 编码query（图像+文本）
    inputs = processor(
        text=query_text,
        images=Image.open(image_path),
        return_tensors="pt"
    )
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
    inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)
    
    try:
        with torch.no_grad():
            qry_output = model(qry=inputs)["qry_reps"]
    except RuntimeError as e:
        # 捕获Flash Attention运行时错误
        if "FlashAttention only supports Ampere" in str(e):
            raise RuntimeError(
                "检测到Flash Attention运行时错误：您的GPU不支持Flash Attention。\n"
                "请在运行脚本前设置环境变量: export USE_FLASH_ATTENTION=0\n"
                f"原始错误: {str(e)}"
            )
        else:
            raise
    
    # 对50个谓词分别计算相似度
    predicate_scores = []
    for predicate in PREDICATES:
        inputs = processor(text=predicate, images=None, return_tensors="pt")
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        
        with torch.no_grad():
            tgt_output = model(tgt=inputs)["tgt_reps"]
            similarity = model.compute_similarity(qry_output, tgt_output)
        
        predicate_scores.append({
            'predicate': predicate,
            'similarity': similarity.item()
        })
    
    return predicate_scores


def calculate_recall_at_k_per_image(image_candidate_predictions, k=50):

    # 按相似度排序，取top-k
    predictions_sorted = sorted(image_candidate_predictions, key=lambda x: x['similarity'], reverse=True)
    top_k_predictions = predictions_sorted[:k]
    
    # 统计top-k中预测正确的关系（去重，每个关系只算一次）
    recalled_relations = set()
    for pred in top_k_predictions:
        if pred['is_correct']:
            recalled_relations.add(pred['relation_idx'])
    
    # 总GT关系数（从候选中提取唯一的relation_idx数量）
    total_gt_relations = len(set(pred['relation_idx'] for pred in image_candidate_predictions))
    
    recall = len(recalled_relations) / total_gt_relations if total_gt_relations > 0 else 0.0
    
    return {
        'recall@k': recall,
        'k': k,
        'recalled_relations': len(recalled_relations),
        'total_gt_relations': total_gt_relations,
        'total_candidates': len(image_candidate_predictions),
        'top_k_candidates': len(top_k_predictions)
    }


def calculate_average_recall_at_k(per_image_candidates, k=50):

    per_image_results = []
    total_recall = 0.0
    valid_images = 0
    
    for image_id, candidates in per_image_candidates.items():
        # 计算该图片的recall
        img_result = calculate_recall_at_k_per_image(candidates, k)
        img_result['image_id'] = image_id
        per_image_results.append(img_result)
        
        total_recall += img_result['recall@k']
        valid_images += 1
    
    # 计算平均recall
    avg_recall = total_recall / valid_images if valid_images > 0 else 0.0
    
    # 统计总体信息
    total_gt_relations = sum(r['total_gt_relations'] for r in per_image_results)
    total_recalled_relations = sum(r['recalled_relations'] for r in per_image_results)
    
    return {
        'avg_recall@k': avg_recall,
        'k': k,
        'total_images': valid_images,
        'total_gt_relations': total_gt_relations,
        'total_recalled_relations': total_recalled_relations,
        'per_image_results': per_image_results
    }



def main():
    print("="*80)
    print("场景图关系预测与Per-Image Recall@50计算")
    print("="*80)

    # 加载数据
    print(f"\n📖 正在加载数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    total_images = len(data)
    total_relations = sum(len(img['relations']) for img in data)
    print(f"   加载了 {total_images} 张图片，共 {total_relations} 个关系")
    
    #  加载模型
    print("\n🔧 正在加载VLM2Vec模型...")
    

    model_args = ModelArguments(
        model_name='/public/home/xiaojw2025/Workspace/VLM2Vec/models/qwen_vl/Qwen2-VL-2B-Instruct',
        checkpoint_path='/public/home/xiaojw2025/Workspace/VLM2Vec/models/VLM2Vec-Qwen2VL-2B',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )
    
    data_args = DataArguments(
        resize_min_pixels=56 * 56,
        resize_max_pixels=28 * 28 * 1280
    )
    
    processor = load_processor(model_args, data_args)
    
    # 尝试加载模型，如果flash attention失败则降级
    try:
        model = MMEBModel.load(model_args)
        model = model.to('cuda', dtype=torch.bfloat16)
        model.eval()
        print("   ✅ 模型加载完成")
    except Exception as e:
        error_msg = str(e)
        # 检查是否是Flash Attention相关错误
        if ("flash" in error_msg.lower() or 
            "ampere" in error_msg.lower() or 
            "attention" in error_msg.lower() and "support" in error_msg.lower()):
            print(f"\n⚠️  模型加载/运行失败: {error_msg[:200]}")
            print("   检测到Flash Attention兼容性问题")
            print("   尝试降级到eager模式...")
            
            # 强制使用eager模式（通过环境变量）
            os.environ["ATTN_IMPLEMENTATION"] = "eager"
            os.environ["USE_FLASH_ATTENTION"] = "0"
            
            # 需要重新导入模块以应用新的环境变量
            import importlib
            import src.model.model
            importlib.reload(src.model.model)
            from src.model.model import MMEBModel as MMEBModelReloaded
            
            try:
                # 重新加载处理器和模型
                processor = load_processor(model_args, data_args)
                model = MMEBModelReloaded.load(model_args)
                model = model.to('cuda', dtype=torch.bfloat16)
                model.eval()
                print("   ✅ 模型加载完成 (使用eager模式)")
            except Exception as e2:
                print(f"\n❌ 降级后仍然失败: {e2}")
                raise
        else:
            print(f"\n❌ 模型加载失败: {error_msg}")
            raise
    
    # 3. 批量预测
    print("\n🚀 开始批量预测...\n")
    
    per_image_candidates = {}  # 按图片组织的候选预测 {image_id: [candidates]}
    all_relations_info = []  # 每个关系的详细信息
    
    global_relation_idx = 0  # 全局关系索引
    
    for img_idx, img_data in enumerate(tqdm(data, desc="处理图片")):
        image_id = img_data['image_id']
        image_path = img_data['image_path']
        objects = img_data['objects']
        relations = img_data['relations']
        
        # 检查图像是否存在
        if not os.path.exists(image_path):
            print(f"⚠️  警告: 图像不存在 {image_path}")
            continue
        
        # 获取图像尺寸
        with Image.open(image_path) as img:
            original_width, original_height = img.size
        
        # 创建物体ID到物体信息的映射
        obj_dict = {obj['id']: obj for obj in objects}
        
        # 初始化该图片的候选列表
        image_candidates = []
        image_relation_idx = 0  # 该图片内的关系索引
        
        # 对每个关系进行预测
        for rel_idx, relation in enumerate(relations):
            subject_id = relation['subject_id']
            object_id = relation['object_id']
            gt_predicate = relation['predicate']
            
            subject_obj = obj_dict[subject_id]
            object_obj = obj_dict[object_id]
            
            # 预测50个谓词的相似度
            predicate_scores = predict_relation(
                model, processor, image_path,
                subject_obj, object_obj,
                original_width, original_height
            )
            
            # 记录该关系的信息
            all_relations_info.append({
                'relation_idx': global_relation_idx,
                'image_id': image_id,
                'image_relation_idx': image_relation_idx,
                'subject': subject_obj['class_name'],
                'object': object_obj['class_name'],
                'gt_predicate': gt_predicate
            })
            
            # 将该关系的50个谓词候选加入该图片的候选池
            for pred_score in predicate_scores:
                image_candidates.append({
                    'relation_idx': image_relation_idx,  # 使用图片内的关系索引
                    'global_relation_idx': global_relation_idx,
                    'image_id': image_id,
                    'subject': subject_obj['class_name'],
                    'object': object_obj['class_name'],
                    'gt_predicate': gt_predicate,
                    'predicted_predicate': pred_score['predicate'],
                    'similarity': pred_score['similarity'],
                    'is_correct': (pred_score['predicate'] == gt_predicate)
                })
            
            image_relation_idx += 1
            global_relation_idx += 1
        
        # 保存该图片的所有候选
        per_image_candidates[image_id] = image_candidates
    
    print(f"\n✅ 预测完成！")
    print(f"   总图片数: {len(per_image_candidates)}")
    print(f"   总关系数: {len(all_relations_info)}")
    total_candidates = sum(len(candidates) for candidates in per_image_candidates.values())
    print(f"   总候选预测数: {total_candidates}")
    
    # 4. 计算Per-Image Recall@50并取平均
    print("\n📊 计算Per-Image Recall@50（每张图片独立计算再平均）...")
    recall_results = calculate_average_recall_at_k(per_image_candidates, k=50)
    
    print("\n" + "="*80)
    print("评估结果 (Per-Image Recall@50)")
    print("="*80)
    print(f"平均 Recall@{recall_results['k']}: {recall_results['avg_recall@k']:.4f} ({recall_results['avg_recall@k']*100:.2f}%)")
    print(f"总图片数: {recall_results['total_images']}")
    print(f"总召回关系数: {recall_results['total_recalled_relations']}/{recall_results['total_gt_relations']}")
    print("="*80)
    
    # 5. 显示每张图片的recall分布
    per_image_recalls = [r['recall@k'] for r in recall_results['per_image_results']]
    if per_image_recalls:
        print(f"\nRecall分布统计:")
        print(f"  最大值: {max(per_image_recalls):.4f}")
        print(f"  最小值: {min(per_image_recalls):.4f}")
        print(f"  中位数: {sorted(per_image_recalls)[len(per_image_recalls)//2]:.4f}")
    
    # 6. 收集所有候选用于展示（可选）
    all_candidate_predictions = []
    for candidates in per_image_candidates.values():
        all_candidate_predictions.extend(candidates)
    
    candidates_sorted = sorted(all_candidate_predictions, key=lambda x: x['similarity'], reverse=True)
    top50_global_candidates = candidates_sorted[:50]
    
    # 7. 保存结果
    print(f"\n💾 正在保存结果到: {OUTPUT_FILE}")
    output_data = {
        'summary': {
            'evaluation_method': 'per-image',
            'total_images': len(per_image_candidates),
            'total_relations': len(all_relations_info),
            'total_candidates': total_candidates,
            'avg_recall@50': recall_results['avg_recall@k'],
            'total_recalled_relations': recall_results['total_recalled_relations'],
            'total_gt_relations': recall_results['total_gt_relations']
        },
        'per_image_results': recall_results['per_image_results'],
        'all_relations': all_relations_info,
        'top50_global_candidates': top50_global_candidates,  # 全局排序的top50（参考用）
        # 'all_candidates': all_candidate_predictions  # 完整的候选列表（可选，可能很大）
    }
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("✅ 结果已保存！")
    
    # 8. 显示一些样例
    print("\n" + "="*80)
    print("Per-Image Recall样例（前5张图片）")
    print("="*80)
    for i, img_result in enumerate(recall_results['per_image_results'][:5], 1):
        print(f"\n{i}. 图片#{img_result['image_id']}")
        print(f"   Recall@50: {img_result['recall@k']:.4f} ({img_result['recall@k']*100:.2f}%)")
        print(f"   召回: {img_result['recalled_relations']}/{img_result['total_gt_relations']} 关系")
        print(f"   候选数: {img_result['total_candidates']} (Top-{img_result['k']}中取{img_result['top_k_candidates']}个)")
    
    print("\n" + "="*80)
    print("全局Top-50候选预测样例（前10个，仅供参考）")
    print("="*80)
    for i, pred in enumerate(top50_global_candidates[:10], 1):
        status = "✅" if pred['is_correct'] else "❌"
        print(f"\n{i}. {status} 排名#{i} (相似度: {pred['similarity']:.4f})")
        print(f"   图片#{pred['image_id']}, 关系#{pred['image_relation_idx']}: {pred['subject']} --[{pred['predicted_predicate']}]--> {pred['object']}")
        print(f"   GT谓词: {pred['gt_predicate']}")


if __name__ == "__main__":
    main()


    