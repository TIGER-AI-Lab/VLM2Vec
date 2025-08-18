import datasets
from datasets import load_dataset
from src.utils.basic_utils import print_rank
import os


def sample_dataset(dataset, **kwargs):
    dataset_name = kwargs.get("dataset_name", "UNKNOWN-DATASET")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", None)

    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if type(num_sample_per_subset) is int and num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print_rank(f"Subsample {dataset_name} to {len(dataset)} samples")

    return dataset


def load_qrels_mapping(qrels):
    """
    Returns:
        {
            "qid1": {"docA": 2, "docB": 1},
            "qid2": {"docC": 3},
            ...
        }
    """
    qrels_mapping = {}

    for row in qrels:
        qid = row["query-id"]
        docid = row["corpus-id"]
        score = row["score"]

        if score > 0:
            if qid not in qrels_mapping:
                qrels_mapping[qid] = {}
            # keep the higher score if already exists
            existing_score = qrels_mapping[qid].get(docid, 0)
            qrels_mapping[qid][docid] = max(existing_score, score)

    return qrels_mapping


def load_hf_dataset(hf_path):
    repo, subset, split = hf_path
    if subset and split:
        return load_dataset(repo, subset, split=split)
    elif subset:
        return load_dataset(repo, subset)
    elif split:
        return load_dataset(repo, split=split)
    else:
        return load_dataset(repo)


def load_local_hf_dataset(dataset_path: str, subset: str = None, split: str = None):
    """
    Loads a Hugging Face dataset from local Parquet files.
    Args:
        dataset_path (str): The base path to the dataset directory
        subset (str, optional): The name of the subdirectory containing the data files (e.g., "corpus").
        split (str, optional): Which split of the data to load (e.g., "train", "test").
    Returns: Dataset or DatasetDict: The loaded dataset.
    """
    if subset and split:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)
    elif subset:
        dataset = datasets.load_dataset(dataset_path, subset)
    elif split:
        dataset = datasets.load_dataset(dataset_path, split=split)
    else:
        dataset = datasets.load_dataset(dataset_path)
    return dataset


def load_hf_dataset_multiple_subset(hf_path, subset_names):
    """
    Load and concatenate multiple subsets from a Hugging Face dataset (e.g. MVBench)
    """
    repo, _, split = hf_path
    subsets = []
    for subset_name in subset_names:
        dataset = load_dataset(repo, subset_name, split=split)
        new_column = [subset_name] * len(dataset)
        dataset = dataset.add_column("subset", new_column)
        subsets.append(dataset)
    dataset = datasets.concatenate_datasets(subsets)
    return dataset

