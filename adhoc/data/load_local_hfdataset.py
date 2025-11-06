from src.constant.dataset_hflocal_path import EVAL_DATASET_HF_PATH
from src.utils.dataset_utils import load_local_hf_dataset

if __name__ == "__main__":
    print("\n" + "="*20 + " Testing Local HF Dataset Loading " + "="*20)
    for dataset_name, (local_path, subset, split) in EVAL_DATASET_HF_PATH.items():
        if "image-tasks" not in local_path:
            continue
        # print(f"\n--- Attempting to load dataset: {dataset_name} ---")
        # print(f"  Path: {local_path}, Subset: {subset}, Split: {split}")
        try:
            loaded_dataset = load_local_hf_dataset(dataset_path=local_path, subset=subset, split=split)
            print(f"  Successfully loaded {dataset_name}, Total samples: {len(loaded_dataset)}")
        except Exception as e:
            print(f"  Failed to load {dataset_name}: {e}")
            raise e

    print("\n" + "="*20 + " Local HF Dataset Loading Test Complete " + "="*20)
