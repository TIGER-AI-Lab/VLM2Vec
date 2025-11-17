"""
评估场景图关系预测结果的脚本
用于分析 predict_scene_graph_recall.py 输出的 JSON 文件
"""

import json
import argparse
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple


def load_results(json_path: str) -> Dict:
    """加载预测结果JSON文件"""
    print(f"📖 正在加载结果文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ 加载完成\n")
    return data


def calculate_mean_rank(data: Dict) -> Dict:
    """
    计算Mean Rank (MR) 指标
    
    对于每个GT关系，找到正确谓词在所有候选中的排名，然后求平均
    """
    print("📊 计算 Mean Rank (MR) 指标...")
    
    # 优先使用 per_image_top100_candidates，如果没有则使用 all_candidates
    all_candidates = []
    
    if 'per_image_top100_candidates' in data:
        print("   使用 per_image_top100_candidates 字段...")
        for image_id, candidates in data['per_image_top100_candidates'].items():
            all_candidates.extend(candidates)
    elif 'all_candidates' in data:
        print("   使用 all_candidates 字段...")
        all_candidates = data['all_candidates']
    else:
        print("⚠️  JSON中没有保存候选列表 (per_image_top100_candidates 或 all_candidates 字段缺失)")
        print("   无法计算 Mean Rank\n")
        return None
    
    # 按 (image_id, relation_idx) 分组（只统计有效的GT关系，排除-1）
    relation_candidates = defaultdict(list)
    for cand in all_candidates:
        if cand['relation_idx'] >= 0:  # 只统计有效的GT关系
            key = (cand['image_id'], cand['relation_idx'])
            relation_candidates[key].append(cand)
    
    ranks = []
    rank_distribution = defaultdict(int)
    
    for key, candidates in relation_candidates.items():
        # 按相似度排序
        sorted_candidates = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
        
        # 找到正确预测的排名（第一个正确的位置），排除no relation预测
        correct_rank = None
        for rank, cand in enumerate(sorted_candidates, 1):
            if cand['is_correct'] and cand.get('predicted_predicate') != 'no relation':
                correct_rank = rank
                break
        
        if correct_rank is not None:
            ranks.append(correct_rank)
            rank_distribution[correct_rank] += 1
    
    mean_rank = np.mean(ranks) if ranks else 0.0
    median_rank = np.median(ranks) if ranks else 0.0
    
    # 计算 MRR (Mean Reciprocal Rank)
    reciprocal_ranks = [1.0 / r for r in ranks]
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    print(f"✅ Mean Rank (MR): {mean_rank:.2f}")
    print(f"✅ Median Rank: {median_rank:.2f}")
    print(f"✅ Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"   统计了 {len(ranks)} 个关系的排名\n")
    
    return {
        'mean_rank': mean_rank,
        'median_rank': median_rank,
        'mrr': mrr,
        'total_relations': len(ranks),
        'rank_distribution': dict(sorted(rank_distribution.items()))
    }


def calculate_recall_at_multiple_k(data: Dict, k_values: List[int] = [1, 5, 10, 20, 50, 100]) -> Dict:
    """
    计算多个K值下的Recall@K
    
    注意：这需要候选列表 (per_image_top100_candidates 或 all_candidates)
    修改：先过滤no relation，再取top-k，然后只保留GT中存在的关系对进行评估（PredCls设置）
    """
    print(f"📊 计算多个K值下的 Recall@K (先过滤no relation，再取top-k): {k_values}")
    
    # 优先使用 per_image_top100_candidates
    per_image_candidates = {}
    
    if 'per_image_top100_candidates' in data:
        print("   使用 per_image_top100_candidates 字段...")
        per_image_candidates = data['per_image_top100_candidates']
        # 检查K值是否超过100
        max_k = max(k_values)
        if max_k > 100:
            print(f"⚠️  警告: K值 {max_k} 超过了保存的Top-100候选，结果可能不准确")
    elif 'all_candidates' in data:
        print("   使用 all_candidates 字段...")
        all_candidates = data['all_candidates']
        # 按 image_id 分组
        per_image_candidates_list = defaultdict(list)
        for cand in all_candidates:
            per_image_candidates_list[cand['image_id']].append(cand)
        per_image_candidates = dict(per_image_candidates_list)
    else:
        print("⚠️  JSON中没有保存候选列表，无法计算不同K值的Recall\n")
        return None
    
    results = {}
    
    for k in k_values:
        total_recall = 0.0
        valid_images = 0
        total_gt_relations = 0
        total_recalled_relations = 0
        
        for image_id, candidates in per_image_candidates.items():
            # 获取该图片中所有GT关系（只统计relation_idx >= 0的，排除-1）
            gt_relations = set()
            for cand in candidates:
                if cand['relation_idx'] >= 0:  # 只统计有效的GT关系
                    gt_relations.add(cand['relation_idx'])
            
            # 第一步：过滤掉no relation的预测（从所有候选中）
            non_bg_candidates = []
            for cand in candidates:
                if cand.get('predicted_predicate') != 'no relation':
                    non_bg_candidates.append(cand)
            
            # 第二步：按相似度排序，取top-k
            sorted_candidates = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)
            top_k = sorted_candidates[:min(k, len(sorted_candidates))]
            
            # 第三步：在top-k中，只对GT关系对进行评估，统计召回的关系（去重）
            recalled_relations = set()
            for cand in top_k:
                # 只统计GT关系对中预测正确的
                if cand['relation_idx'] in gt_relations and cand['is_correct']:
                    recalled_relations.add(cand['relation_idx'])
            
            # 计算该图片的recall
            total_gt_in_image = len(gt_relations)
            recalled_in_image = len(recalled_relations)
            
            recall = recalled_in_image / total_gt_in_image if total_gt_in_image > 0 else 0.0
            total_recall += recall
            valid_images += 1
            total_gt_relations += total_gt_in_image
            total_recalled_relations += recalled_in_image
        
        avg_recall = total_recall / valid_images if valid_images > 0 else 0.0
        overall_recall = total_recalled_relations / total_gt_relations if total_gt_relations > 0 else 0.0
        
        results[f'recall@{k}'] = avg_recall
        results[f'overall_recall@{k}'] = overall_recall
        results[f'stats@{k}'] = {
            'total_gt_relations': total_gt_relations,
            'total_recalled_relations': total_recalled_relations,
            'valid_images': valid_images
        }
        
        print(f"   Recall@{k:3d}: {avg_recall:.4f} ({avg_recall*100:.2f}%) [平均]")
        print(f"   Overall@{k:3d}: {overall_recall:.4f} ({overall_recall*100:.2f}%) [整体]")
        print(f"   统计@{k:3d}: {total_recalled_relations}/{total_gt_relations} 关系被召回，{valid_images} 张图片")
    
    print()
    return results


def calculate_category_recall_at_k(data: Dict, k_values: List[int] = [1, 5, 10, 20, 50, 100]) -> Dict:
    """
    计算base和novel谓词类别的Recall@K
    
    Args:
        data: 预测结果数据
        k_values: K值列表
    
    Returns:
        字典，包含base和novel类别的recall统计
    修改：先过滤no relation，再取top-k，然后只保留GT中存在的关系对进行评估（PredCls设置）
    """
    print(f"📊 计算Base和Novel谓词类别的Recall@K (先过滤no relation，再取top-k): {k_values}")
    
    # 加载谓词类别映射
    predicate_category_mapping = {
        "above": "base", "across": "novel", "against": "base", "along": "novel", "and": "novel",
        "at": "base", "attached to": "base", "behind": "base", "belonging to": "base", "between": "base",
        "carrying": "base", "covered in": "base", "covering": "base", "eating": "novel", "flying in": "novel",
        "for": "base", "from": "base", "growing on": "novel", "hanging from": "base", "has": "base",
        "holding": "base", "in": "base", "in front of": "base", "laying on": "novel", "looking at": "base",
        "lying on": "novel", "made of": "base", "mounted on": "novel", "near": "base", "of": "base",
        "on": "base", "on back of": "novel", "over": "base", "painted on": "novel", "parked on": "base",
        "part of": "novel", "playing": "base", "riding": "base", "says": "novel", "sitting on": "base",
        "standing on": "base", "to": "base", "under": "base", "using": "novel", "walking in": "novel",
        "walking on": "base", "watching": "base", "wearing": "base", "wears": "base", "with": "base"
    }
    
    # 优先使用 per_image_top100_candidates
    per_image_candidates = {}
    
    if 'per_image_top100_candidates' in data:
        print("   使用 per_image_top100_candidates 字段...")
        per_image_candidates = data['per_image_top100_candidates']
        max_k = max(k_values)
        if max_k > 100:
            print(f"⚠️  警告: K值 {max_k} 超过了保存的Top-100候选，结果可能不准确")
    elif 'all_candidates' in data:
        print("   使用 all_candidates 字段...")
        all_candidates = data['all_candidates']
        per_image_candidates_list = defaultdict(list)
        for cand in all_candidates:
            per_image_candidates_list[cand['image_id']].append(cand)
        per_image_candidates = dict(per_image_candidates_list)
    else:
        print("⚠️  JSON中没有保存候选列表，无法计算类别recall\n")
        return None
    
    results = {}
    
    for k in k_values:
        # 初始化每个谓词的统计（按谓词分类）
        predicate_stats = {}
        for pred_name, category in predicate_category_mapping.items():
            predicate_stats[pred_name] = {'hit': 0, 'total': 0, 'category': category}
        
        for image_id, candidates in per_image_candidates.items():
            # 获取该图片中所有GT关系（只统计relation_idx >= 0的，排除-1）
            gt_relations = set()
            for cand in candidates:
                if cand['relation_idx'] >= 0:  # 只统计有效的GT关系
                    gt_relations.add(cand['relation_idx'])
            
            # 第一步：过滤掉no relation的预测（从所有候选中）
            non_bg_candidates = []
            for cand in candidates:
                if cand.get('predicted_predicate') != 'no relation':
                    non_bg_candidates.append(cand)
            
            # 第二步：按相似度排序，取top-k
            predictions_sorted = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)
            actual_k = min(k, len(predictions_sorted))
            top_k_predictions = predictions_sorted[:actual_k]
            
            # 统计该图片中每个谓词的GT
            gt_predicates_in_image = {}
            recalled_predicates_in_image = {}
            
            for cand in candidates:
                gt_pred = cand.get('gt_predicate')
                if gt_pred is None or gt_pred not in predicate_stats:
                    continue
                relation_idx = cand['relation_idx']
                
                # 统计GT（每个关系只算一次，只统计有效的GT关系，排除-1）
                if relation_idx >= 0 and relation_idx not in gt_predicates_in_image:
                    gt_predicates_in_image[relation_idx] = gt_pred
                    predicate_stats[gt_pred]['total'] += 1
            
            # 第三步：在top-k中，只对GT关系对进行评估，统计召回的谓词
            for cand in top_k_predictions:
                # 只统计GT关系对中预测正确的
                if cand['relation_idx'] in gt_relations and cand['is_correct']:
                    relation_idx = cand['relation_idx']
                    gt_pred = cand.get('gt_predicate')
                    if gt_pred is None or gt_pred not in predicate_stats:
                        continue
                    
                    # 每个关系只算一次召回
                    if relation_idx not in recalled_predicates_in_image:
                        recalled_predicates_in_image[relation_idx] = gt_pred
                        predicate_stats[gt_pred]['hit'] += 1
        
        # 计算每个谓词的recall，然后按类别分组平均
        base_recalls = []
        novel_recalls = []
        base_total = 0
        novel_total = 0
        base_hit = 0
        novel_hit = 0
        
        for pred, stats in predicate_stats.items():
            if stats['total'] > 0:
                pred_recall = stats['hit'] / stats['total']
                if stats['category'] == 'base':
                    base_recalls.append(pred_recall)
                    base_total += stats['total']
                    base_hit += stats['hit']
                elif stats['category'] == 'novel':
                    novel_recalls.append(pred_recall)
                    novel_total += stats['total']
                    novel_hit += stats['hit']
        
        # 计算每个类别的mean recall（简单平均，不加权）
        category_recall = {}
        category_recall['base'] = {
            'recall': np.mean(base_recalls) if base_recalls else 0.0,
            'hit': base_hit,
            'total': base_total,
            'num_predicates': len(base_recalls)
        }
        category_recall['novel'] = {
            'recall': np.mean(novel_recalls) if novel_recalls else 0.0,
            'hit': novel_hit,
            'total': novel_total,
            'num_predicates': len(novel_recalls)
        }
        
        results[f'category_recall@{k}'] = category_recall
        
        # 打印结果
        base_recall = category_recall['base']['recall']
        novel_recall = category_recall['novel']['recall']
        base_info = f"{category_recall['base']['hit']}/{category_recall['base']['total']}, {category_recall['base']['num_predicates']}谓词"
        novel_info = f"{category_recall['novel']['hit']}/{category_recall['novel']['total']}, {category_recall['novel']['num_predicates']}谓词"
        
        print(f"   Recall@{k:3d} - Base: {base_recall:.4f} ({base_recall*100:.2f}%) [{base_info}]")
        print(f"   Recall@{k:3d} - Novel: {novel_recall:.4f} ({novel_recall*100:.2f}%) [{novel_info}]")
    
    print()
    return results


def calculate_mean_recall_per_predicate_multi_k(data: Dict, k_values: List[int] = [1, 5, 10, 20, 50, 100]) -> Dict:
    """
    计算多个K值下每个谓词类别的Mean Recall
    
    Args:
        data: 预测结果数据
        k_values: K值列表
    
    Returns:
        字典，包含每个K值下的谓词MR统计
    修改：先过滤no relation，再取top-k，然后只保留GT中存在的关系对进行评估（PredCls设置）
    """
    print(f"📊 计算多个K值下的谓词级别 Mean Recall (先过滤no relation，再取top-k): {k_values}")
    
    # 优先使用 per_image_top100_candidates
    per_image_candidates = {}
    
    if 'per_image_top100_candidates' in data:
        print("   使用 per_image_top100_candidates 字段...")
        per_image_candidates = data['per_image_top100_candidates']
        max_k = max(k_values)
        if max_k > 100:
            print(f"⚠️  警告: K值 {max_k} 超过了保存的Top-100候选，结果可能不准确")
    elif 'all_candidates' in data:
        print("   使用 all_candidates 字段...")
        all_candidates = data['all_candidates']
        per_image_candidates_list = defaultdict(list)
        for cand in all_candidates:
            per_image_candidates_list[cand['image_id']].append(cand)
        per_image_candidates = dict(per_image_candidates_list)
    else:
        print("⚠️  JSON中没有保存候选列表，无法计算谓词MR\n")
        return None
    
    # 获取所有谓词列表（过滤None值）
    predicates_set = set()
    for candidates in per_image_candidates.values():
        for cand in candidates:
            predicate = cand.get('gt_predicate')
            if predicate is not None:  # 过滤None值
                predicates_set.add(predicate)
    predicates = sorted(list(predicates_set))
    
    results = {}
    
    for k in k_values:
        # 初始化每个谓词的统计
        predicate_stats = {pred: {'hit': 0, 'total': 0} for pred in predicates}
        
        for image_id, candidates in per_image_candidates.items():
            # 获取该图片中所有GT关系（只统计relation_idx >= 0的，排除-1）
            gt_relations = set()
            for cand in candidates:
                if cand['relation_idx'] >= 0:  # 只统计有效的GT关系
                    gt_relations.add(cand['relation_idx'])
            
            # 第一步：过滤掉no relation的预测（从所有候选中）
            non_bg_candidates = []
            for cand in candidates:
                if cand.get('predicted_predicate') != 'no relation':
                    non_bg_candidates.append(cand)
            
            # 第二步：按相似度排序，取top-k
            predictions_sorted = sorted(non_bg_candidates, key=lambda x: x['similarity'], reverse=True)
            actual_k = min(k, len(predictions_sorted))
            top_k_predictions = predictions_sorted[:actual_k]
            
            # 统计该图片中每个谓词类别的GT
            gt_predicates_in_image = {}
            recalled_predicates_in_image = {}
            
            for cand in candidates:
                gt_pred = cand.get('gt_predicate')
                if gt_pred is None:  # 跳过None值
                    continue
                relation_idx = cand['relation_idx']
                
                # 统计GT（每个关系只算一次，只统计有效的GT关系，排除-1）
                if relation_idx >= 0 and relation_idx not in gt_predicates_in_image:
                    gt_predicates_in_image[relation_idx] = gt_pred
                    predicate_stats[gt_pred]['total'] += 1
            
            # 第三步：在top-k中，只对GT关系对进行评估，统计召回的谓词
            for cand in top_k_predictions:
                # 只统计GT关系对中预测正确的
                if cand['relation_idx'] in gt_relations and cand['is_correct']:
                    relation_idx = cand['relation_idx']
                    gt_pred = cand.get('gt_predicate')
                    if gt_pred is None:  # 跳过None值
                        continue
                    
                    # 每个关系只算一次召回
                    if relation_idx not in recalled_predicates_in_image:
                        recalled_predicates_in_image[relation_idx] = gt_pred
                        predicate_stats[gt_pred]['hit'] += 1
        
        # 计算每个谓词的recall
        per_predicate_recall = {}
        valid_recalls = []
        
        for pred in predicates:
            total = predicate_stats[pred]['total']
            hit = predicate_stats[pred]['hit']
            
            if total > 0:
                recall = hit / total
                per_predicate_recall[pred] = {
                    'recall': recall,
                    'hit': hit,
                    'total': total
                }
                valid_recalls.append(recall)
            else:
                per_predicate_recall[pred] = {
                    'recall': 0.0,
                    'hit': 0,
                    'total': 0
                }
        
        # 计算mean recall（只对有GT的类别计算）
        mean_recall = np.mean(valid_recalls) if valid_recalls else 0.0
        
        results[f'mean_recall@{k}'] = {
            'mean_recall': mean_recall,
            'num_valid_predicates': len(valid_recalls),
            'total_predicates': len(predicates),
            'per_predicate_recall': per_predicate_recall
        }
        
        print(f"   Mean Recall@{k:3d}: {mean_recall:.4f} ({mean_recall*100:.2f}%), 有效谓词: {len(valid_recalls)}/{len(predicates)}")
    
    print()
    return results


def display_category_recall_results(category_recall_results: Dict) -> None:
    """
    展示Base和Novel谓词类别的Recall结果
    
    Args:
        category_recall_results: 类别recall结果
    """
    if not category_recall_results:
        return
    
    print("="*80)
    print("📊 Base和Novel谓词类别Recall详细分析")
    print("="*80)
    
    # 提取所有k值并排序
    k_values = sorted([int(k.split('@')[1]) for k in category_recall_results.keys()])
    
    # 打印总体趋势
    print("\n📈 Base vs Novel Recall对比:")
    print(f"{'K值':<10}{'Base Recall':<15}{'Novel Recall':<15}{'Base谓词数':<15}{'Novel谓词数':<15}")
    print("-"*70)
    
    for k in k_values:
        key = f'category_recall@{k}'
        base_recall = category_recall_results[key]['base']['recall']
        novel_recall = category_recall_results[key]['novel']['recall']
        base_num = category_recall_results[key]['base']['num_predicates']
        novel_num = category_recall_results[key]['novel']['num_predicates']
        
        print(f"K={k:<8}{base_recall:<15.4f}{novel_recall:<15.4f}{base_num:<15}{novel_num:<15}")
    
    # 分析性能差异
    print(f"\n📊 性能差异分析:")
    for k in k_values:
        key = f'category_recall@{k}'
        base_recall = category_recall_results[key]['base']['recall']
        novel_recall = category_recall_results[key]['novel']['recall']
        
        if base_recall > 0 and novel_recall > 0:
            ratio = base_recall / novel_recall
            diff = base_recall - novel_recall
            
            print(f"K={k}: Base/Novel比率 = {ratio:.2f}, 差异 = {diff:+.4f}")
            
            if ratio > 1.2:
                print(f"   → Base谓词表现明显更好")
            elif ratio < 0.8:
                print(f"   → Novel谓词表现明显更好")
            else:
                print(f"   → 两类谓词表现相当")
    
    # 统计信息
    print(f"\n📈 统计信息:")
    base_nums = [category_recall_results[f'category_recall@{k}']['base']['num_predicates'] for k in k_values]
    novel_nums = [category_recall_results[f'category_recall@{k}']['novel']['num_predicates'] for k in k_values]
    base_totals = [category_recall_results[f'category_recall@{k}']['base']['total'] for k in k_values]
    novel_totals = [category_recall_results[f'category_recall@{k}']['novel']['total'] for k in k_values]
    
    print(f"Base谓词类别数: {max(base_nums) if base_nums else 0}")
    print(f"Novel谓词类别数: {max(novel_nums) if novel_nums else 0}")
    print(f"Base关系实例数: {max(base_totals) if base_totals else 0}")
    print(f"Novel关系实例数: {max(novel_totals) if novel_totals else 0}")
    
    if base_nums and novel_nums:
        base_novel_ratio = max(base_nums) / max(novel_nums) if max(novel_nums) > 0 else 0
        print(f"Base/Novel谓词数量比: {base_novel_ratio:.2f}")
    
    print()


def display_predicate_multi_k_results(predicate_multi_k_results: Dict, top_n: int = 10, detail_k_values: List[int] = None) -> None:
    """
    展示多K值下的谓词级别Mean Recall结果
    
    Args:
        predicate_multi_k_results: 多K值下的谓词MR结果
        top_n: 显示Top-N和Bottom-N的谓词
        detail_k_values: 需要显示详细谓词信息的K值列表，None表示只显示总体趋势
    """
    if not predicate_multi_k_results:
        return
    
    print("="*80)
    print("📊 多K值下的谓词级别 Mean Recall 详细分析")
    print("="*80)
    
    # 提取所有k值并排序
    k_values = sorted([int(k.split('@')[1]) for k in predicate_multi_k_results.keys()])
    
    # 打印总体Mean Recall趋势
    print("\n📈 总体Mean Recall趋势:")
    print(f"{'K值':<10}{'Mean Recall':<15}{'百分比':<12}{'有效谓词数':<15}")
    print("-"*52)
    for k in k_values:
        key = f'mean_recall@{k}'
        mr = predicate_multi_k_results[key]['mean_recall']
        valid = predicate_multi_k_results[key]['num_valid_predicates']
        total = predicate_multi_k_results[key]['total_predicates']
        print(f"K={k:<8}{mr:<15.4f}{mr*100:<12.2f}{valid}/{total}")
    
        # 如果未指定detail_k_values，则默认显示所有K值的详细信息
        if detail_k_values is None:
            # 显示所有K值
            detail_k_values = k_values
    
    # 只为指定的K值显示详细的Top-N和Bottom-N谓词
    for k in detail_k_values:
        if k not in k_values:
            continue
            
        key = f'mean_recall@{k}'
        per_predicate = predicate_multi_k_results[key]['per_predicate_recall']
        
        # 过滤有GT的谓词并排序
        predicates_with_gt = [(pred, stats) for pred, stats in per_predicate.items() if stats['total'] > 0]
        sorted_predicates = sorted(predicates_with_gt, key=lambda x: x[1]['recall'], reverse=True)
        
        print(f"\n{'='*80}")
        print(f"📊 K={k} 时的谓词性能详情")
        print(f"{'='*80}")
        
        print(f"\n📋 所有谓词性能详情 (共 {len(sorted_predicates)} 个谓词):")
        print(f"{'排名':<6}{'谓词':<25}{'Recall':<12}{'命中/总数':<15}")
        print("-"*60)
        for i, (pred, stats) in enumerate(sorted_predicates, 1):
            print(f"{i:<6}{pred:<25}{stats['recall']:<12.4f}{stats['hit']}/{stats['total']}")
    
    print()


def analyze_predicate_performance(data: Dict) -> None:
    """分析每个谓词类别的性能"""
    print("="*80)
    print("📊 谓词类别性能分析 (基于预测时的K值)")
    print("="*80)
    
    per_predicate = data.get('mean_recall_per_predicate', {})
    
    if not per_predicate:
        print("⚠️  没有找到谓词级别的统计信息\n")
        return
    
    # 按recall排序
    sorted_predicates = sorted(
        per_predicate.items(),
        key=lambda x: x[1]['recall'],
        reverse=True
    )
    
    # 只显示有GT的谓词
    predicates_with_gt = [(pred, stats) for pred, stats in sorted_predicates if stats['total'] > 0]
    
    if not predicates_with_gt:
        print("⚠️  没有有效的谓词统计\n")
        return
    
    print(f"\n总共 {len(predicates_with_gt)} 个谓词类别有GT数据\n")
    
    # 计算统计信息
    recalls = [stats['recall'] for _, stats in predicates_with_gt]
    mean_recall = np.mean(recalls)
    median_recall = np.median(recalls)
    std_recall = np.std(recalls)
    
    print(f"谓词Recall统计:")
    print(f"  平均值: {mean_recall:.4f}")
    print(f"  中位数: {median_recall:.4f}")
    print(f"  标准差: {std_recall:.4f}")
    print(f"  最大值: {max(recalls):.4f}")
    print(f"  最小值: {min(recalls):.4f}")
    
    # 显示所有谓词（不限制数量）
    print("\n📋 所有谓词性能排名（按Recall排序）:")
    print(f"{'排名':<6}{'谓词':<25}{'Recall':<10}{'命中/总数':<15}")
    print("-"*60)
    for i, (pred, stats) in enumerate(predicates_with_gt, 1):
        print(f"{i:<6}{pred:<25}{stats['recall']:<10.4f}{stats['hit']}/{stats['total']:<15}")
    
    print()


def analyze_image_performance(data: Dict) -> None:
    """分析每张图片的性能分布"""
    print("="*80)
    print("📊 图片级别性能分析")
    print("="*80)
    
    per_image_results = data.get('per_image_results', [])
    
    if not per_image_results:
        print("⚠️  没有找到图片级别的统计信息\n")
        return
    
    recalls = [img['recall@k'] for img in per_image_results]
    gt_relations = [img['total_gt_relations'] for img in per_image_results]
    recalled = [img['recalled_relations'] for img in per_image_results]
    
    print(f"\n总图片数: {len(per_image_results)}")
    print(f"\nRecall@K 分布统计:")
    print(f"  平均值: {np.mean(recalls):.4f}")
    print(f"  中位数: {np.median(recalls):.4f}")
    print(f"  标准差: {np.std(recalls):.4f}")
    print(f"  最大值: {np.max(recalls):.4f}")
    print(f"  最小值: {np.min(recalls):.4f}")
    
    # 百分位数
    print(f"\nRecall@K 百分位数:")
    for percentile in [25, 50, 75, 90, 95]:
        value = np.percentile(recalls, percentile)
        print(f"  {percentile}th: {value:.4f}")
    
    # 按recall分组统计
    print(f"\nRecall@K 分组统计:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins)-1):
        count = sum(1 for r in recalls if bins[i] <= r < bins[i+1])
        percentage = count / len(recalls) * 100
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:4d} 张图片 ({percentage:5.2f}%)")
    
    # 显示表现最好和最差的图片
    sorted_images = sorted(per_image_results, key=lambda x: x['recall@k'], reverse=True)
    
    print("\n🏆 Top-20 表现最好的图片:")
    print(f"{'图片ID':<15}{'Recall@K':<12}{'召回关系':<15}{'总关系':<10}")
    print("-"*60)
    for img in sorted_images[:20]:
        print(f"{str(img['image_id']):<15}{img['recall@k']:<12.4f}"
              f"{img['recalled_relations']:<15}{img['total_gt_relations']:<10}")
    
    print("\n⚠️  Bottom-20 表现最差的图片:")
    print(f"{'图片ID':<15}{'Recall@K':<12}{'召回关系':<15}{'总关系':<10}")
    print("-"*60)
    for img in sorted_images[-20:]:
        print(f"{str(img['image_id']):<15}{img['recall@k']:<12.4f}"
              f"{img['recalled_relations']:<15}{img['total_gt_relations']:<10}")
    
    print()


def analyze_relation_count_distribution(data: Dict) -> None:
    """分析图像关系数量分布"""
    print("="*80)
    print("📊 图像关系数量分布分析")
    print("="*80)
    
    per_image_results = data.get('per_image_results', [])
    
    if not per_image_results:
        print("⚠️  没有找到图片级别的统计信息\n")
        return
    
    # 收集所有图片的关系数量
    relation_counts = [img['total_gt_relations'] for img in per_image_results]
    
    # 基本统计
    print(f"\n📈 关系数量基本统计:")
    print(f"  总图片数: {len(relation_counts)}")
    print(f"  平均关系数: {np.mean(relation_counts):.2f}")
    print(f"  中位数关系数: {np.median(relation_counts):.2f}")
    print(f"  标准差: {np.std(relation_counts):.2f}")
    print(f"  最小值: {np.min(relation_counts)}")
    print(f"  最大值: {np.max(relation_counts)}")
    
    # 百分位数
    print(f"\n📊 关系数量百分位数:")
    for percentile in [25, 50, 75, 90, 95, 99]:
        value = np.percentile(relation_counts, percentile)
        print(f"  {percentile}th: {value:.1f}")
    
    # 分布统计
    print(f"\n📊 关系数量分布:")
    bins = [0, 5, 10, 15, 20, 25, 30, 50, 100, float('inf')]
    bin_labels = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-50', '51-100', '100+']
    
    for i in range(len(bins)-1):
        count = sum(1 for c in relation_counts if bins[i] < c <= bins[i+1])
        percentage = count / len(relation_counts) * 100
        print(f"  {bin_labels[i]:<8}: {count:4d} 张图片 ({percentage:5.2f}%)")
    
    # 关系数量频率表
    from collections import Counter
    relation_freq = Counter(relation_counts)
    print(f"\n📊 具体关系数量频率 (全部):")
    print(f"{'关系数':<8}{'图片数':<8}{'百分比':<10}")
    print("-"*30)
    for num_rel, count in relation_freq.most_common():
        percentage = count / len(relation_counts) * 100
        print(f"{num_rel:<8}{count:<8}{percentage:<10.2f}")
    
    print()


def analyze_relation_count_impact(data: Dict) -> None:
    """分析图片中关系数量对Recall的影响"""
    print("="*80)
    print("📊 关系数量对Recall的影响分析")
    print("="*80)
    
    per_image_results = data.get('per_image_results', [])
    
    if not per_image_results:
        print("⚠️  没有找到图片级别的统计信息\n")
        return
    
    # 按关系数量分组
    relation_groups = defaultdict(list)
    for img in per_image_results:
        num_relations = img['total_gt_relations']
        relation_groups[num_relations].append(img['recall@k'])
    
    # 计算每组的平均recall
    group_stats = []
    for num_rel, recalls in sorted(relation_groups.items()):
        avg_recall = np.mean(recalls)
        median_recall = np.median(recalls)
        group_stats.append({
            'num_relations': num_rel,
            'avg_recall': avg_recall,
            'median_recall': median_recall,
            'num_images': len(recalls),
            'std_recall': np.std(recalls),
            'min_recall': np.min(recalls),
            'max_recall': np.max(recalls)
        })
    
    print(f"\n📈 按关系数量分组的Recall性能:")
    print(f"{'关系数':<8}{'图片数':<8}{'平均Recall':<12}{'中位数Recall':<15}{'标准差':<10}{'范围':<15}")
    print("-"*80)
    for stat in group_stats:
        recall_range = f"{stat['min_recall']:.3f}-{stat['max_recall']:.3f}"
        print(f"{stat['num_relations']:<8}{stat['num_images']:<8}"
              f"{stat['avg_recall']:<12.4f}{stat['median_recall']:<15.4f}"
              f"{stat['std_recall']:<10.4f}{recall_range:<15}")
    
    # 按关系数量区间分析
    print(f"\n📊 按关系数量区间的性能分析:")
    relation_ranges = [
        (1, 5, "1-5个关系"),
        (6, 10, "6-10个关系"), 
        (11, 15, "11-15个关系"),
        (16, 20, "16-20个关系"),
        (21, 30, "21-30个关系"),
        (31, 50, "31-50个关系"),
        (51, 100, "51-100个关系"),
        (101, float('inf'), "100+个关系")
    ]
    
    print(f"{'关系数范围':<15}{'图片数':<8}{'平均Recall':<12}{'中位数Recall':<15}{'标准差':<10}")
    print("-"*70)
    
    for min_rel, max_rel, label in relation_ranges:
        range_images = [img for img in per_image_results 
                       if min_rel <= img['total_gt_relations'] <= max_rel]
        
        if len(range_images) > 0:
            recalls = [img['recall@k'] for img in range_images]
            avg_recall = np.mean(recalls)
            median_recall = np.median(recalls)
            std_recall = np.std(recalls)
            
            print(f"{label:<15}{len(range_images):<8}{avg_recall:<12.4f}"
                  f"{median_recall:<15.4f}{std_recall:<10.4f}")
    
    # 相关性分析
    relation_counts = [img['total_gt_relations'] for img in per_image_results]
    recalls = [img['recall@k'] for img in per_image_results]
    
    correlation = np.corrcoef(relation_counts, recalls)[0, 1]
    print(f"\n📈 关系数量与Recall的相关性:")
    print(f"  皮尔逊相关系数: {correlation:.4f}")
    
    if correlation > 0.1:
        print("  → 关系数量与Recall呈正相关")
    elif correlation < -0.1:
        print("  → 关系数量与Recall呈负相关")
    else:
        print("  → 关系数量与Recall相关性较弱")
    
    print()


def analyze_detailed_relation_performance(data: Dict) -> None:
    """详细分析不同关系数量下的性能表现"""
    print("="*80)
    print("📊 详细关系数量性能分析")
    print("="*80)
    
    per_image_results = data.get('per_image_results', [])
    
    if not per_image_results:
        print("⚠️  没有找到图片级别的统计信息\n")
        return
    
    # 按关系数量分组，更细致的分析
    relation_groups = defaultdict(list)
    for img in per_image_results:
        num_relations = img['total_gt_relations']
        relation_groups[num_relations].append({
            'image_id': img.get('image_id', 'unknown'),
            'recall': img['recall@k'],
            'recalled_relations': img.get('recalled_relations', 0),
            'total_gt_relations': img['total_gt_relations']
        })
    
    # 分析每个关系数量下的性能
    print(f"\n📈 各关系数量下的详细性能:")
    print(f"{'关系数':<8}{'图片数':<8}{'平均Recall':<12}{'中位数Recall':<15}{'标准差':<10}{'最佳Recall':<12}{'最差Recall':<12}")
    print("-"*90)
    
    for num_rel in sorted(relation_groups.keys()):
        group_data = relation_groups[num_rel]
        recalls = [item['recall'] for item in group_data]
        
        avg_recall = np.mean(recalls)
        median_recall = np.median(recalls)
        std_recall = np.std(recalls)
        best_recall = np.max(recalls)
        worst_recall = np.min(recalls)
        
        print(f"{num_rel:<8}{len(group_data):<8}{avg_recall:<12.4f}"
              f"{median_recall:<15.4f}{std_recall:<10.4f}"
              f"{best_recall:<12.4f}{worst_recall:<12.4f}")
    
    # 按关系数量区间进行更细致的分析
    print(f"\n📊 按关系数量区间的详细分析:")
    
    # 定义更细致的区间
    detailed_ranges = [
        (1, 3, "1-3个关系"),
        (4, 6, "4-6个关系"),
        (7, 10, "7-10个关系"),
        (11, 15, "11-15个关系"),
        (16, 20, "16-20个关系"),
        (21, 25, "21-25个关系"),
        (26, 30, "26-30个关系"),
        (31, 40, "31-40个关系"),
        (41, 50, "41-50个关系"),
        (51, 75, "51-75个关系"),
        (76, 100, "76-100个关系"),
        (101, float('inf'), "100+个关系")
    ]
    
    print(f"{'关系数范围':<15}{'图片数':<8}{'平均Recall':<12}{'中位数Recall':<15}{'标准差':<10}{'最佳':<8}{'最差':<8}")
    print("-"*90)
    
    for min_rel, max_rel, label in detailed_ranges:
        range_images = [img for img in per_image_results 
                       if min_rel <= img['total_gt_relations'] <= max_rel]
        
        if len(range_images) > 0:
            recalls = [img['recall@k'] for img in range_images]
            avg_recall = np.mean(recalls)
            median_recall = np.median(recalls)
            std_recall = np.std(recalls)
            best_recall = np.max(recalls)
            worst_recall = np.min(recalls)
            
            print(f"{label:<15}{len(range_images):<8}{avg_recall:<12.4f}"
                  f"{median_recall:<15.4f}{std_recall:<10.4f}"
                  f"{best_recall:<8.4f}{worst_recall:<8.4f}")
    
    # 找出性能最好和最差的关系数量
    print(f"\n🏆 性能分析总结:")
    
    # 计算每个关系数量的平均recall
    relation_performance = {}
    for num_rel, group_data in relation_groups.items():
        recalls = [item['recall'] for item in group_data]
        relation_performance[num_rel] = {
            'avg_recall': np.mean(recalls),
            'count': len(recalls),
            'std': np.std(recalls)
        }
    
    # 按平均recall排序
    sorted_performance = sorted(relation_performance.items(), 
                              key=lambda x: x[1]['avg_recall'], reverse=True)
    
    print(f"\n📈 关系数量性能排名 (全部):")
    print(f"{'排名':<6}{'关系数':<8}{'图片数':<8}{'平均Recall':<12}{'标准差':<10}")
    print("-"*50)
    for i, (num_rel, stats) in enumerate(sorted_performance, 1):
        print(f"{i:<6}{num_rel:<8}{stats['count']:<8}"
              f"{stats['avg_recall']:<12.4f}{stats['std']:<10.4f}")
    
    # 分析性能趋势
    print(f"\n📊 性能趋势分析:")
    if len(sorted_performance) >= 3:
        best_relation_count = sorted_performance[0][0]
        worst_relation_count = sorted_performance[-1][0]
        
        print(f"  最佳性能关系数: {best_relation_count} (Recall: {sorted_performance[0][1]['avg_recall']:.4f})")
        print(f"  最差性能关系数: {worst_relation_count} (Recall: {sorted_performance[-1][1]['avg_recall']:.4f})")
        
        # 分析是否存在明显趋势
        relation_counts = [item[0] for item in sorted_performance]
        avg_recalls = [item[1]['avg_recall'] for item in sorted_performance]
        
        # 计算关系数量与recall的相关性
        correlation = np.corrcoef(relation_counts, avg_recalls)[0, 1]
        print(f"  关系数量与性能相关性: {correlation:.4f}")
        
        if correlation > 0.3:
            print("  → 关系数量越多，性能越好")
        elif correlation < -0.3:
            print("  → 关系数量越多，性能越差")
        else:
            print("  → 关系数量与性能没有明显线性关系")
    
    print()


def print_summary(data: Dict) -> None:
    """打印总结信息"""
    print("="*80)
    print("📋 总结报告")
    print("="*80)
    
    summary = data.get('summary', {})
    
    if not summary:
        print("⚠️  没有找到总结信息\n")
        return
    
    print(f"\n评估方法: {summary.get('evaluation_method', 'N/A')}")
    print(f"总图片数: {summary.get('total_images', 0)}")
    print(f"总关系数: {summary.get('total_relations', 0)}")
    print(f"总候选数: {summary.get('total_candidates', 0)}")
    
    print(f"\n主要指标:")
    print(f"  Average Recall@50: {summary.get('avg_recall@50', 0):.4f} ({summary.get('avg_recall@50', 0)*100:.2f}%)")
    print(f"  Mean Recall@50:    {summary.get('mean_recall@50', 0):.4f} ({summary.get('mean_recall@50', 0)*100:.2f}%)")
    
    print(f"\n召回统计:")
    print(f"  总召回关系数: {summary.get('total_recalled_relations', 0)}")
    print(f"  总GT关系数:   {summary.get('total_gt_relations', 0)}")
    if summary.get('total_gt_relations', 0) > 0:
        overall_recall = summary.get('total_recalled_relations', 0) / summary.get('total_gt_relations', 1)
        print(f"  整体召回率:   {overall_recall:.4f} ({overall_recall*100:.2f}%)")
    
    print(f"\n谓词统计:")
    print(f"  有效谓词类别数: {summary.get('num_valid_predicates', 0)}")
    
    if summary.get('images_with_insufficient_candidates', 0) > 0:
        print(f"\n⚠️  注意:")
        print(f"  有 {summary.get('images_with_insufficient_candidates', 0)} 张图片的候选数不足50")
    
    print()


def export_detailed_report(data: Dict, output_path: str, additional_metrics: Dict = None) -> None:
    """导出详细的评估报告"""
    print(f"💾 正在导出详细报告到: {output_path}")
    
    report = {
        'summary': data.get('summary', {}),
        'additional_metrics': additional_metrics or {},
        'per_image_statistics': {
            'total_images': len(data.get('per_image_results', [])),
            'recall_distribution': {}
        },
        'per_predicate_statistics': {
            'total_predicates': len([p for p, s in data.get('mean_recall_per_predicate', {}).items() if s['total'] > 0]),
        }
    }
    
    # 添加图片级别统计
    if data.get('per_image_results'):
        recalls = [img['recall@k'] for img in data['per_image_results']]
        report['per_image_statistics']['recall_distribution'] = {
            'mean': float(np.mean(recalls)),
            'median': float(np.median(recalls)),
            'std': float(np.std(recalls)),
            'min': float(np.min(recalls)),
            'max': float(np.max(recalls)),
            'percentiles': {
                '25th': float(np.percentile(recalls, 25)),
                '50th': float(np.percentile(recalls, 50)),
                '75th': float(np.percentile(recalls, 75)),
                '90th': float(np.percentile(recalls, 90)),
                '95th': float(np.percentile(recalls, 95)),
            }
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 报告已导出\n")


def main():
    parser = argparse.ArgumentParser(
        description="评估场景图关系预测结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本评估
  python evaluate_results.py results.json
  
  # 包含Mean Rank计算（需要完整候选列表）
  python evaluate_results.py results.json --calculate-mr
  
  # 计算多个K值的Recall
  python evaluate_results.py results.json --multi-k
  
  # 计算多个K值下的谓词级别Mean Recall
  python evaluate_results.py results.json --predicate-multi-k
  
  # 计算Base和Novel谓词类别的Recall
  python evaluate_results.py results.json --category-recall
  
  # 自定义K值列表
  python evaluate_results.py results.json --multi-k --predicate-multi-k --category-recall --k-values 1 5 10 20 50
  
  # 导出详细报告
  python evaluate_results.py results.json --export report.json
  
  # 完整分析
  python evaluate_results.py results.json --calculate-mr --multi-k --predicate-multi-k --category-recall --export report.json
        """
    )
    
    parser.add_argument('--json_file', type=str, default='/public/home/xiaojw2025/Workspace/VLM2Vec/predict/recall_results_2000_qwen2vl_2b_instruct.json', help='预测结果JSON文件路径')
    parser.add_argument('--calculate-mr', action='store_true', 
                       help='计算Mean Rank (MR) 指标（需要完整候选列表）')
    parser.add_argument('--multi-k', action='store_true',
                       help='计算多个K值下的Recall@K（需要完整候选列表）')
    parser.add_argument('--predicate-multi-k', action='store_true',
                       help='计算多个K值下的谓词级别Mean Recall（需要完整候选列表）')
    parser.add_argument('--category-recall', action='store_true',
                       help='计算Base和Novel谓词类别的Recall@K（需要完整候选列表）')
    parser.add_argument('--k-values', type=int, nargs='+', default=[50, 100],
                       help='指定要计算的K值列表（默认: 1 5 10 20 50 100）')
    parser.add_argument('--export', type=str, default=None,
                       help='导出详细报告到指定JSON文件')
    parser.add_argument('--no-predicate-analysis', action='store_true',
                       help='跳过谓词级别分析')
    parser.add_argument('--no-image-analysis', action='store_true',
                       help='跳过图片级别分析')
    parser.add_argument('--detailed-relation-analysis', action='store_true',
                       help='进行详细的关系数量性能分析')
    parser.add_argument('--no-relation-distribution', action='store_true',
                       help='跳过关系数量分布分析')
    
    args = parser.parse_args()
    
    # 加载结果
    data = load_results(args.json_file)
    
    # 打印总结
    print_summary(data)
    
    # 额外的指标
    additional_metrics = {}
    
    # 计算Mean Rank（默认启用）
    mr_results = calculate_mean_rank(data)
    if mr_results:
        additional_metrics['mean_rank_metrics'] = mr_results
    
    # 计算多个K值的Recall（默认启用）
    multi_k_results = calculate_recall_at_multiple_k(data, args.k_values)
    if multi_k_results:
        additional_metrics['multi_k_recall'] = multi_k_results
    
    # 计算多个K值下的谓词级别Mean Recall（默认启用）
    predicate_multi_k_results = calculate_mean_recall_per_predicate_multi_k(data, args.k_values)
    if predicate_multi_k_results:
        additional_metrics['predicate_mean_recall_multi_k'] = predicate_multi_k_results
    
    # 计算Base和Novel谓词类别的Recall（默认启用）
    category_recall_results = calculate_category_recall_at_k(data, args.k_values)
    if category_recall_results:
        additional_metrics['category_recall'] = category_recall_results
    
    # 谓词级别分析（默认启用，除非明确禁用）
    if not args.no_predicate_analysis:
        analyze_predicate_performance(data)
    
    # 展示多K值下的谓词MR详细结果（显示全部，所有K值）
    if predicate_multi_k_results:
        display_predicate_multi_k_results(predicate_multi_k_results, top_n=9999, detail_k_values=args.k_values)
    
    # 展示Base和Novel谓词类别的Recall结果
    if category_recall_results:
        display_category_recall_results(category_recall_results)
    
    # 图片级别分析（默认启用，除非明确禁用）
    if not args.no_image_analysis:
        analyze_image_performance(data)
    
    # 关系数量分布分析（默认启用，除非明确禁用）
    if not args.no_relation_distribution:
        analyze_relation_count_distribution(data)
    
    # 关系数量影响分析（默认启用）
    analyze_relation_count_impact(data)
    
    # 详细关系数量性能分析（默认启用）
    analyze_detailed_relation_performance(data)
    
    # 导出报告
    if args.export:
        export_detailed_report(data, args.export, additional_metrics)
    
    print("="*80)
    print("✅ 评估完成！")
    print("="*80)


if __name__ == "__main__":
    main()

