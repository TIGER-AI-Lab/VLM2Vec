from datasets.distributed import split_dataset_by_node
from datasets import Sequence, Value, Dataset as HFDataset, IterableDataset as HFIterableDataset

from src.data.dataset.base_pair_dataset import AutoPairDataset
from src.data.dataset.hf_datasets import interleave_datasets
from src.utils.basic_utils import print_master
import torch


def _safe_num_rows(dataset):
    num_rows = getattr(dataset, "num_rows", None)
    if isinstance(num_rows, int):
        return num_rows
    try:
        return len(dataset)
    except (TypeError, NotImplementedError):
        return None


def init_mixed_dataset(dataset_config, model_args, data_args, training_args):
    weights = [d['weight'] for d in dataset_config.values()]
    w_sum = sum(weights)
    probs = [w / w_sum for w in weights]
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    train_datasets = []
    image_feature = {
        "paths": Sequence(Value("string")),
        "bytes": Sequence(Value("binary")),
        "resolutions": Sequence(Sequence(Value("int32"), length=2)),
    }
    audio_feature = {
        "path": Value("string"),
        "bytes": Value("binary"),
        "start": Value("float32"),
        "end": Value("float32"),
    }

    for data_idx, (global_dataset_name, dataset_config) in enumerate(dataset_config.items()):
        train_dataset = AutoPairDataset.instantiate(model_args=model_args, data_args=data_args, training_args=training_args, **dataset_config)
        num_rows_before_cast = _safe_num_rows(train_dataset)
        for col, feature in [
            ("query_image", image_feature),
            ("pos_image", image_feature),
            ("neg_image", image_feature),
            ("query_audio", audio_feature),
            ("pos_audio", audio_feature),
        ]:
            if hasattr(train_dataset, "cast_column") and train_dataset.column_names and col in train_dataset.column_names:
                train_dataset = train_dataset.cast_column(col, feature)

        # cast_column may return a new IterableDataset object and drop dynamically-set attributes.
        if getattr(train_dataset, "num_rows", None) is None and isinstance(num_rows_before_cast, int):
            setattr(train_dataset, "num_rows", num_rows_before_cast)
        dataset_num_rows = _safe_num_rows(train_dataset)
        dataset_num_rows_msg = dataset_num_rows if dataset_num_rows is not None else "unknown"
        print_master(
            f"\t\tDataset#{data_idx} (dataset_parser={dataset_config.get('dataset_parser', 'n/a')}): "
            f"{global_dataset_name}, num_rows={dataset_num_rows_msg}, prob={probs[data_idx] * 100.0}"
        )
        train_datasets.append(train_dataset)

    if training_args.interleave_batch_size and training_args.interleave_batch_size <= 1.0:
        interleave_batch_size = training_args.per_device_train_batch_size * world_size * training_args.interleave_batch_size
    else:
        interleave_batch_size = training_args.interleave_batch_size
    dataset_num_rows = [_safe_num_rows(d) for d in train_datasets]
    total_num_rows = (
        sum(dataset_num_rows)
        if all(isinstance(n, int) for n in dataset_num_rows)
        else None
    )
    total_num_rows_msg = total_num_rows if total_num_rows is not None else "unknown"
    estimated_steps = (
        total_num_rows / (training_args.per_device_train_batch_size * world_size)
        if total_num_rows is not None
        else "unknown"
    )
    print_master(f"\nInitializing interleave datasets:"
                 f"\n\t\tworld_size={world_size}"
                 f"\n\t\ttotal num rows={total_num_rows_msg}"
                 f"\n\t\tglobal batch size={training_args.per_device_train_batch_size * world_size}"
                 f"\n\t\testimated num step per epoch={estimated_steps}"
                 f"\n\t\tinterleave_batch_size={interleave_batch_size}"
                 )
    if total_num_rows is not None:
        assert total_num_rows >= (training_args.per_device_train_batch_size * world_size), \
            f"total_num_rows(={total_num_rows}) must be greater than or equal to global batch size (={training_args.per_device_train_batch_size * world_size}), since the last batch will be dropped."

    if len(train_datasets) > 1:
        # HF interleave requires homogeneous input type:
        # all map-style Dataset OR all IterableDataset.
        has_map_style = any(isinstance(d, HFDataset) for d in train_datasets)
        has_iterable = any(isinstance(d, HFIterableDataset) for d in train_datasets)
        if has_map_style and has_iterable:
            num_workers = getattr(training_args, "dataloader_num_workers", 0) or 0
            num_shards = max(world_size, world_size * max(int(num_workers), 1))
            print_master(
                f"Detected mixed dataset types (Dataset + IterableDataset). "
                f"Converting map-style datasets to IterableDataset (num_shards={num_shards})."
            )
            converted = []
            for ds in train_datasets:
                if isinstance(ds, HFDataset):
                    ds_num_rows = _safe_num_rows(ds)
                    ds = ds.to_iterable_dataset(num_shards=num_shards)
                    if isinstance(ds_num_rows, int):
                        setattr(ds, "num_rows", ds_num_rows)
                converted.append(ds)
            train_datasets = converted

        train_dataset = interleave_datasets(train_datasets, probabilities=probs, batch_size=interleave_batch_size,
                                            seed=training_args.seed, stopping_strategy=training_args.interleave_stopping_strategy)
    else:
        train_dataset = train_datasets[0]
    if torch.distributed.is_initialized():
        train_dataset = split_dataset_by_node(train_dataset, rank=torch.distributed.get_rank(), world_size=world_size)

    return train_dataset
