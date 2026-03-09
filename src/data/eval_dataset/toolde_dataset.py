from typing import List
from datasets import Dataset
import os
import json
import math


class DatasetWithLength:
    """包装Dataset对象，确保len()函数可以正常工作"""
    def __init__(self, dataset, num_rows):
        self.dataset = dataset
        self._num_rows = num_rows
    
    def __len__(self):
        return self._num_rows
    
    def __getattr__(self, name):
        # 委托所有其他属性访问到原始数据集
        return getattr(self.dataset, name)
    
    def __iter__(self):
        return iter(self.dataset)
    
    def __getitem__(self, key):
        return self.dataset[key]

DATASET_PARSER_NAME = "toolde"
from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH as EVAL_DATASET_LOCAL_PATH
from src.data.eval_dataset.base_eval_dataset import (
    AutoEvalPairDataset, 
    add_metainfo_hook, 
    RESOLUTION_MAPPING,
    ImageVideoInstance
)
from src.model.processor import PHI3V, VLM_IMAGE_TOKENS


def create_empty_image_dict(image_resolution):
    """创建一个空的图像字典，用于纯文本数据"""
    return {
        "bytes": [None], 
        "paths": [None], 
        "resolutions": [RESOLUTION_MAPPING.get(image_resolution, None)]
    }


@add_metainfo_hook
def data_prepare_query(batch_dict, *args, **kwargs):
    """
    处理ToolRet数据集的query数据
    """
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']
    
    batch_size = len(batch_dict['qry_text'])
    query_texts, query_images, dataset_infos = [], [], []
    
    for qry_text, qry_id, label_names, subtask in \
        zip(batch_dict['qry_text'], 
            batch_dict['qry_id'],
            batch_dict['label_names'],
            batch_dict.get('subtask', [None] * batch_size)):
        
        if not qry_text:
            print("empty query text")
            continue
            
        # 处理不同模型的特殊token
        if model_backbone != PHI3V:
            qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        
        query_texts.append([qry_text])
        
        # 为纯文本数据创建空的图像占位符
        empty_image = create_empty_image_dict(image_resolution)
        query_images.append([empty_image])
        
        # 保存数据集信息
        dataset_infos.append({
            "qry_id": qry_id,
            "label_name": label_names,  # 这是一个列表，包含多个positive doc IDs
            "cand_names": [],  # 在全局检索模式下，这个会是空的
            "subtask": subtask,
        })
    
    if len(query_texts) == 0:
        print('something went wrong in query preparation')
    
    # 为全局检索模式添加空的cand_text和cand_image字段
    # generate_cand_dataset函数期望这些字段存在
    cand_texts = [[] for _ in query_texts]
    cand_images = [[] for _ in query_texts]
    
    return {
        "query_text": query_texts, 
        "query_image": query_images,
        "cand_text": cand_texts,
        "cand_image": cand_images,
        "dataset_infos": dataset_infos
    }


@add_metainfo_hook
def data_prepare_candidate(batch_dict, *args, **kwargs):
    """
    处理ToolRet数据集的candidate数据
    """
    model_backbone = kwargs['model_backbone']
    image_resolution = kwargs['image_resolution']
    
    batch_size = len(batch_dict['cand_text'])
    cand_texts, cand_images, dataset_infos = [], [], []
    
    for cand_text, cand_id in zip(batch_dict['cand_text'], batch_dict['cand_id']):
        if not cand_text:
            print("empty candidate text")
            continue
            
        # 处理不同模型的特殊token
        if model_backbone != PHI3V:
            cand_text = cand_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[model_backbone])
        
        cand_texts.append([cand_text])
        
        # 为纯文本数据创建空的图像占位符
        empty_image = create_empty_image_dict(image_resolution)
        cand_images.append([empty_image])
        
        dataset_infos.append({
            "cand_name": cand_id,
        })
    
    if len(cand_texts) == 0:
        print('something went wrong in candidate preparation')
    
    return {
        "cand_text": cand_texts, 
        "cand_image": cand_images, 
        "dataset_infos": dataset_infos
    }


def load_query_data(query_file_path):
    """
    加载query文件
    支持Parquet格式、JSONL格式和JSON格式
    字段: id, query, label (或 labels), subtask
    """
    import pandas as pd

    queries = []

    if query_file_path.endswith('.parquet'):
        # Parquet格式：每行一个query记录
        df = pd.read_parquet(query_file_path)
        for row in df.to_dict(orient="records"):
            if "subtask" not in row:
                # MMEB-V3 parquet 常见字段是 category
                row["subtask"] = row.get("category", None)
            queries.append(_parse_query_data(row))
    # 检查文件扩展名，支持.json和.jsonl
    elif query_file_path.endswith('.json') and not query_file_path.endswith('.jsonl'):
        # JSON格式：整个文件是一个JSON数组
        with open(query_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            if isinstance(data_list, list):
                for data in data_list:
                    queries.append(_parse_query_data(data))
    else:
        # JSONL格式：每行一个JSON对象
        with open(query_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                queries.append(_parse_query_data(data))
    
    return queries


def _parse_query_data(data):
    """
    解析单个query数据项
    字段: id, query, label (或 labels), subtask
    """
    instruction = data.get("instruction", "")
    query_text = data.get("query", "")
    if instruction:
        query_text = f"{instruction} {query_text}".strip()

    # 支持 label (单数) 和 labels (复数) 字段
    label_data = data.get('label', None)
    if label_data is None:
        label_data = data.get('labels', [])
    # parquet 中可能出现 NaN，按空处理
    if isinstance(label_data, float) and math.isnan(label_data):
        label_data = []
    pos_ids = []

    try:
        # label/labels可能是JSON字符串、列表或其他格式
        if isinstance(label_data, str):
            # 尝试解析JSON字符串
            labels = json.loads(label_data)
        else:
            labels = label_data
        
        # 处理列表格式
        if isinstance(labels, list):
            for label in labels:
                if isinstance(label, dict) and 'id' in label:
                    pos_ids.append(label['id'])
                elif isinstance(label, str):
                    pos_ids.append(label)
        # 处理单个值
        elif isinstance(labels, dict) and 'id' in labels:
            pos_ids.append(labels['id'])
        elif isinstance(labels, str):
            pos_ids.append(labels)
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，尝试其他格式
        pass

    return {
        'qry_id': data.get('id', ''),
        'qry_text': query_text,
        'label_names': pos_ids,
        'subtask': data.get('subtask', None),
    }


def load_candidate_data(candidate_file_path):
    """
    加载candidate文件
    支持Parquet格式和JSONL格式
    字段: id, documentation
    """
    import pandas as pd

    candidates = []

    if candidate_file_path.endswith('.parquet'):
        # 加载parquet文件
        df = pd.read_parquet(candidate_file_path)
        for _, row in df.iterrows():
            doc_text = row.get('documentation', '')

            # documentation字段可能是JSON字符串，使用完整内容作为候选文本
            try:
                doc_json = json.loads(doc_text) if isinstance(doc_text, str) else doc_text
                if isinstance(doc_json, dict):
                    cand_text = json.dumps(doc_json, ensure_ascii=True)
                else:
                    cand_text = str(doc_json)
            except (json.JSONDecodeError, TypeError):
                cand_text = str(doc_text)

            candidates.append({
                'cand_id': row.get('id', ''),
                'cand_text': cand_text,
            })
    else:
        # JSONL格式：每行一个JSON对象
        with open(candidate_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                
                # 字段: id, documentation
                cand_id = data.get('id', '')
                doc_text = data.get('documentation', '')
                
                # documentation可能是JSON字符串，使用完整内容作为候选文本
                try:
                    doc_json = json.loads(doc_text) if isinstance(doc_text, str) else doc_text
                    if isinstance(doc_json, dict):
                        cand_text = json.dumps(doc_json, ensure_ascii=True)
                    else:
                        cand_text = str(doc_json)
                except (json.JSONDecodeError, TypeError):
                    cand_text = str(doc_text)
                
                candidates.append({
                    'cand_id': cand_id,
                    'cand_text': cand_text,
                })
    return candidates


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_toolde_dataset(model_args, data_args, *args, **kwargs):
    """
    加载ToolDe评估数据集
    
    数据结构:
        - query按子任务分类：web/apigen, code/xxx, customized/xxx
        - candidate按大类分类：web, code, customized
        - 评估时按类别检索：web的query只在web的candidate中检索
    
    参数:
        dataset_name: 数据集名称，默认为"toolde"
        subset_name: 子任务名称，格式为"category/task"，例如"web/apigen"
        query_file: query文件路径
        candidate_file: candidate文件路径（按类别，如web、code、customized）
        data_path: 数据目录路径（可选）
    """
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    subset_name = kwargs.get("subset_name")
    
    # 从subset_name提取类别（如"web/apigen" -> "web"）和任务名（如"web/apigen" -> "apigen"）
    if subset_name and '/' in subset_name:
        category, task_name = subset_name.split('/', 1)
    else:
        category = subset_name
        task_name = subset_name
    
    # 获取数据文件路径
    query_file = kwargs.get("query_file")
    candidate_file = kwargs.get("candidate_file")
    data_path = kwargs.get("data_path", None)
    
    # 如果提供了data_path，构建完整路径
    if data_path:
        if query_file and not os.path.isabs(query_file):
            query_file = os.path.join(data_path, query_file)
        if candidate_file and not os.path.isabs(candidate_file):
            candidate_file = os.path.join(data_path, candidate_file)
    
    # 处理query_file：如果是目录，根据subset_name构建文件路径
    if query_file and os.path.isdir(query_file):
        query_dir = query_file  # 保存原始目录路径
        # 尝试多种可能的文件路径
        possible_query_paths = [
            os.path.join(query_dir, f"{subset_name}.jsonl"),  # web/apigen.jsonl
            os.path.join(query_dir, f"{task_name}.jsonl"),     # apigen.jsonl
            os.path.join(query_dir, category, f"{task_name}.jsonl"),  # web/apigen.jsonl
            os.path.join(query_dir, f"{subset_name}.json"),   # web/apigen.json
            os.path.join(query_dir, f"{task_name}.json"),      # apigen.json
        ]
        query_file = None
        for path in possible_query_paths:
            if os.path.exists(path) and os.path.isfile(path):
                query_file = path
                break
        
        # 如果预设路径都不存在，在目录中搜索匹配的文件
        if query_file is None:
            import glob
            # 搜索包含task_name的文件
            search_patterns = [
                os.path.join(query_dir, f"*{task_name}*.jsonl"),
                os.path.join(query_dir, f"*{task_name}*.json"),
                os.path.join(query_dir, category, f"*{task_name}*.jsonl"),
                os.path.join(query_dir, category, f"*{task_name}*.json"),
            ]
            for pattern in search_patterns:
                matches = glob.glob(pattern)
                if matches:
                    query_file = matches[0]
                    break
        
        if query_file is None:
            raise FileNotFoundError(f"Query file not found in directory {query_dir}. Tried patterns: {possible_query_paths}")
    elif query_file and not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")
    
    # 处理candidate_file：如果是目录，根据category构建文件路径
    if candidate_file and os.path.isdir(candidate_file):
        # 尝试多种可能的文件路径
        possible_candidate_paths = [
            os.path.join(candidate_file, f"{category}.parquet"),  # web.parquet
            os.path.join(candidate_file, "candidates.parquet"),   # candidates.parquet
            os.path.join(candidate_file, f"{category}.jsonl"),    # web.jsonl
            os.path.join(candidate_file, "candidates.jsonl"),     # candidates.jsonl
        ]
        candidate_file_path = None
        for path in possible_candidate_paths:
            if os.path.exists(path) and os.path.isfile(path):
                candidate_file_path = path
                break
        
        # 如果预设路径都不存在，在目录中搜索匹配的文件
        if candidate_file_path is None:
            import glob
            # 搜索parquet或jsonl文件
            search_patterns = [
                os.path.join(candidate_file, "*.parquet"),
                os.path.join(candidate_file, "*.jsonl"),
                os.path.join(candidate_file, "*.json"),
            ]
            for pattern in search_patterns:
                matches = glob.glob(pattern)
                if matches:
                    candidate_file_path = matches[0]
                    break
        
        if candidate_file_path is None:
            raise FileNotFoundError(f"Candidate file not found in directory {candidate_file}. Tried patterns: {possible_candidate_paths}")
        candidate_file = candidate_file_path
    elif candidate_file and not os.path.exists(candidate_file):
        raise FileNotFoundError(f"Candidate file not found: {candidate_file}")
    
    print(f"Loading queries from: {query_file}")
    print(f"Loading candidates from: {candidate_file}")
    print(f"Category: {category}, Subset: {subset_name}")
    
    # 加载数据
    query_data = load_query_data(query_file)
    candidate_data = load_candidate_data(candidate_file)
    
    print(f"Loaded {len(query_data)} queries and {len(candidate_data)} candidates for category '{category}'")
    
    # 创建query数据集
    qry_dataset = Dataset.from_list(query_data)
    num_rows = len(query_data)
    
    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    kwargs['global_dataset_name'] = f'{DATASET_PARSER_NAME}/{subset_name}'
    
    # 处理query数据
    qry_dataset = qry_dataset.map(
        lambda x: data_prepare_query(x, **kwargs), 
        batched=True, 
        batch_size=64,
        remove_columns=['qry_text', 'qry_id', 'label_names', 'subtask'], 
        drop_last_batch=False
    )
    
    # 确保数据集支持len()函数
    # 如果map操作后len()不工作，使用包装类
    try:
        _ = len(qry_dataset)
    except (TypeError, AttributeError):
        # len()不支持，使用包装类
        qry_dataset = DatasetWithLength(qry_dataset, num_rows)
    # 如果len()支持，就不需要做任何操作，Dataset对象已经有num_rows属性了
    
    # 创建corpus（所有候选文档）
    corpus_rows = []
    for cand in candidate_data:
        empty_image = create_empty_image_dict(data_args.image_resolution)
        corpus_rows.append({
            "cand_text": [cand['cand_text']],
            "cand_image": [empty_image],
            "dataset_infos": {
                "cand_names": [cand['cand_id']],
            }
        })
    
    corpus = Dataset.from_list(corpus_rows)
    
    print(f"Created query dataset with {len(query_data)} samples")
    print(f"Created corpus with {len(corpus_rows)} candidates")
    
    return qry_dataset, corpus
