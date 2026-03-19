import os
from datasets import load_dataset
from src.utils.basic_utils import print_rank

# --- Visual Document Datasets ---
VISDOC_DATASETS = {
    '''
    # visdoc datasets follow the BEIR format (corpus, queries, qrels)
    # Visdoc-ViDoRe v1 10
    "ViDoRe_arxivqa": ("vidore/arxivqa_test_subsampled_beir", None, "test"),
    "ViDoRe_docvqa": ("vidore/docvqa_test_subsampled_beir", None, "test"),
    "ViDoRe_infovqa": ("vidore/infovqa_test_subsampled_beir", None, "test"),
    "ViDoRe_tabfquad": ("vidore/tabfquad_test_subsampled_beir", None, "test"),
    "ViDoRe_tatdqa": ("vidore/tatdqa_test_beir", None, "test"),
    "ViDoRe_shiftproject": ("vidore/shiftproject_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_artificial_intelligence": ("vidore/syntheticDocQA_artificial_intelligence_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_energy": ("vidore/syntheticDocQA_energy_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_government_reports": ("vidore/syntheticDocQA_government_reports_test_beir", None, "test"),
    "ViDoRe_syntheticDocQA_healthcare_industry": ("vidore/syntheticDocQA_healthcare_industry_test_beir", None, "test"),
    # Visdoc-ViDoRe v2 4
    "ViDoRe_esg_reports_human_labeled_v2": ("vidore/esg_reports_human_labeled_v2", None, "test"),
    "ViDoRe_biomedical_lectures_v2_multilingual": ("vidore/biomedical_lectures_v2", None, "test"),
    "ViDoRe_economics_reports_v2_multilingual": ("vidore/economics_reports_v2", None, "test"),
    "ViDoRe_esg_reports_v2_multilingual": ("vidore/esg_reports_v2", None, "test"),
    # Visdoc-VisRAG 6
    "VisRAG_ArxivQA": ("openbmb/VisRAG-Ret-Test-ArxivQA", None, "train"),
    "VisRAG_ChartQA": ("openbmb/VisRAG-Ret-Test-ChartQA", None, "train"),
    "VisRAG_MP-DocVQA": ("openbmb/VisRAG-Ret-Test-MP-DocVQA", None, "train"),
    "VisRAG_SlideVQA": ("openbmb/VisRAG-Ret-Test-SlideVQA", None, "train"),
    "VisRAG_InfoVQA": ("openbmb/VisRAG-Ret-Test-InfoVQA", None, "train"),
    "VisRAG_PlotQA": ("openbmb/VisRAG-Ret-Test-PlotQA", None, "train"),
    # VisDoc-OOD 4
    "ViDoSeek-doc": ("VLM2Vec/ViDoSeek", None, "test"),
    # "ViDoSeek-page": ("VLM2Vec/ViDoSeek-page", None, "test"),
    "MMLongBench-doc": ("VLM2Vec/MMLongBench-doc", None, "test"),
    # "MMLongBench-page": ("VLM2Vec/MMLongBench", None, "test"),
    '''
    
    "ViDoSeek-page": ("VLM2Vec/ViDoSeek-page-fixed", None, "test"),
    "MMLongBench-page": ("VLM2Vec/MMLongBench-page-fixed", None, "test"),
}

# --- Video Datasets ---
VIDEO_DATASETS = {
    # Video-RET 5
    "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
    "MSVD": ("VLM2Vec/MSVD", None, "test"),
    "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
    "YouCook2": ("VLM2Vec/YouCook2", None, "val"),
    "VATEX": ("VLM2Vec/VATEX", None, "test"),
    # Video-CLS 5
    "HMDB51": ("VLM2Vec/HMDB51", None, "test"),
    "UCF101": ("VLM2Vec/UCF101", None, "test"),
    "Breakfast": ("VLM2Vec/Breakfast", None, "test"),
    "Kinetics-700": ("VLM2Vec/Kinetics-700", None, "test"),
    "SmthSmthV2": ("VLM2Vec/SmthSmthV2", None, "test"),
    # Video-MR 3
    "QVHighlight": ("VLM2Vec/QVHighlight", None, "test"),
    "Charades-STA": ("VLM2Vec/Charades-STA", None, "test"),
    "MomentSeeker": ("VLM2Vec/MomentSeeker", None, "test"),
    "MomentSeeker_1k8": ("VLM2Vec/MomentSeeker_1k8", None, "test"),
    # Video-QA 5
    "NExTQA": ("VLM2Vec/NExTQA", "MC", "test"),
    "EgoSchema": ("VLM2Vec/egoschema", "Subset", "test"),
    "MVBench": ("VLM2Vec/MVBench/MVBench", None, "train"),
    "Video-MME": ("VLM2Vec/Video-MME", None, "test"),
    "ActivityNetQA": ("VLM2Vec/ActivityNetQA", None, "test"),
}

# --- Image Datasets ---
IMAGE_DATASETS = {
    # Image-CLS 10
    "ImageNet-1K": ("ziyjiang/MMEB_Test_Instruct", "ImageNet-1K", "test"),
    "N24News": ("ziyjiang/MMEB_Test_Instruct", "N24News", "test"),
    "HatefulMemes": ("ziyjiang/MMEB_Test_Instruct", "HatefulMemes", "test"),
    "VOC2007": ("ziyjiang/MMEB_Test_Instruct", "VOC2007", "test"),
    "SUN397": ("ziyjiang/MMEB_Test_Instruct", "SUN397", "test"),
    "Place365": ("ziyjiang/MMEB_Test_Instruct", "Place365", "test"),
    "ImageNet-A": ("ziyjiang/MMEB_Test_Instruct", "ImageNet-A", "test"),
    "ImageNet-R": ("ziyjiang/MMEB_Test_Instruct", "ImageNet-R", "test"),
    "ObjectNet": ("ziyjiang/MMEB_Test_Instruct", "ObjectNet", "test"),
    "Country211": ("ziyjiang/MMEB_Test_Instruct", "Country211", "test"),
    # Image-QA 10
    "OK-VQA": ("ziyjiang/MMEB_Test_Instruct", "OK-VQA", "test"),
    "A-OKVQA": ("ziyjiang/MMEB_Test_Instruct", "A-OKVQA", "test"),
    "DocVQA": ("ziyjiang/MMEB_Test_Instruct", "DocVQA", "test"),
    "InfographicsVQA": ("ziyjiang/MMEB_Test_Instruct", "InfographicsVQA", "test"),
    "ChartQA": ("ziyjiang/MMEB_Test_Instruct", "ChartQA", "test"),
    "Visual7W": ("ziyjiang/MMEB_Test_Instruct", "Visual7W", "test"),
    "ScienceQA": ("ziyjiang/MMEB_Test_Instruct", "ScienceQA", "test"),
    "VizWiz": ("ziyjiang/MMEB_Test_Instruct", "VizWiz", "test"),
    "GQA": ("ziyjiang/MMEB_Test_Instruct", "GQA", "test"),
    "TextVQA": ("ziyjiang/MMEB_Test_Instruct", "TextVQA", "test"),
    # Image-RET 12
    "VisDial": ("ziyjiang/MMEB_Test_Instruct", "VisDial", "test"),
    "CIRR": ("ziyjiang/MMEB_Test_Instruct", "CIRR", "test"),
    "VisualNews_t2i": ("ziyjiang/MMEB_Test_Instruct", "VisualNews_t2i", "test"),
    "VisualNews_i2t": ("ziyjiang/MMEB_Test_Instruct", "VisualNews_i2t", "test"),
    "MSCOCO_t2i": ("ziyjiang/MMEB_Test_Instruct", "MSCOCO_t2i", "test"),
    "MSCOCO_i2t": ("ziyjiang/MMEB_Test_Instruct", "MSCOCO_i2t", "test"),
    "NIGHTS": ("ziyjiang/MMEB_Test_Instruct", "NIGHTS", "test"),
    "WebQA": ("ziyjiang/MMEB_Test_Instruct", "WebQA", "test"),
    "FashionIQ": ("ziyjiang/MMEB_Test_Instruct", "FashionIQ", "test"),
    "Wiki-SS-NQ": ("ziyjiang/MMEB_Test_Instruct", "Wiki-SS-NQ", "test"),
    "OVEN": ("ziyjiang/MMEB_Test_Instruct", "OVEN", "test"),
    "EDIS": ("ziyjiang/MMEB_Test_Instruct", "EDIS", "test"),
    # Image-VG 4
    "MSCOCO": ("ziyjiang/MMEB_Test_Instruct", "MSCOCO", "test"),
    "RefCOCO": ("ziyjiang/MMEB_Test_Instruct", "RefCOCO", "test"),
    "RefCOCO-Matching": ("ziyjiang/MMEB_Test_Instruct", "RefCOCO-Matching", "test"),
    "Visual7W-Pointing": ("ziyjiang/MMEB_Test_Instruct", "Visual7W-Pointing", "test"),
}



def save_dataset_to_disk(dataset, output_path, file_basename=""):
    """Helper function to save a dataset to parquet and jsonl."""
    # Save to parquet
    parquet_path = os.path.join(output_path, f"{file_basename}.parquet")
    df = dataset.to_pandas()
    df.to_parquet(parquet_path)
    print(f"  Saved {file_basename} to {parquet_path}")

    # Save to jsonl
    jsonl_path = os.path.join(output_path, f"{file_basename}.jsonl")
    df.to_json(jsonl_path, orient='records', lines=True)
    print(f"  Saved {file_basename} to {jsonl_path}")

def download_visdoc_datasets(datasets_map, output_dir):
    """
    Downloads visual document datasets, handling the special BEIR format.
    """
    print("\n" + "="*20 + " Downloading Visual Document Datasets " + "="*20)
    for dataset_name, (repo, subset, split) in datasets_map.items():
        print(f"--- Processing {dataset_name} from {repo} ---")
        try:
            for config_name in ['corpus', 'qrels', 'queries']:
                print(f"  Downloading '{dataset_name}-{config_name}'")
                dataset_dir = os.path.join(output_dir, dataset_name, config_name)
                os.makedirs(dataset_dir, exist_ok=True)
                # For BEIR datasets, the config name is passed to the `name` argument
                dataset = load_dataset(repo, name=config_name, split=split)
                save_dataset_to_disk(dataset, dataset_dir, file_basename=split)

        except Exception as e:
            print(f"Failed to download or process {dataset_name}: {e}")

def download_standard_datasets(datasets_map, modality_name, output_dir):
    """Downloads standard (non-BEIR) datasets for a given modality."""
    print(f"\n" + "="*20 + f" Downloading {modality_name.title()} Datasets " + "="*20)
    for name, (repo, subset, split) in datasets_map.items():
        print(f"--- Processing {name} from {repo} ---")
        try:
            dataset_dir = os.path.join(output_dir, name)
            os.makedirs(dataset_dir, exist_ok=True)

            dataset = load_dataset(repo, subset, split=split) if subset else load_dataset(repo, split=split)
            save_dataset_to_disk(dataset, dataset_dir, file_basename=split)

        except Exception as e:
            print(f"Failed to download or process {name}: {e}")


def clone_hf_dataset(repo_id: str, output_path: str):
    """
    Clones a Hugging Face dataset repository using git clone.

    Args:
        repo_id (str): The Hugging Face repository ID (e.g., "vidore/esg_reports_human_labeled_v2").
        output_path (str): The local path where the repository should be cloned.
    """
    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print_rank(f"  Skipping {repo_id}: Directory already exists at {output_path}")
        return

    command = f"git clone {repo_url} {output_path}"
    print_rank(f"  Cloning {repo_id} to {output_path}...")
    print_rank(f"  Executing: {command}")
    
    # Execute the command using run_shell_command
    # The agent will replace this with a tool call.
    # For direct script execution, you might use os.system(command)
    # For the agent, this will be a tool call.
    # Example of how the agent would call it:
    # run_shell_command(command=command, description=f"Cloning {repo_id}")
    
    # Placeholder for agent execution (replace with actual tool call when running as agent)
    # For now, using os.system for script completeness if run directly
    result = os.system(command) # This line will be replaced by the agent with a tool call
    if result != 0:
        print_rank(f"  Failed to clone {repo_id}. Exit code: {result}")
    else:
        print_rank(f"  Successfully cloned {repo_id}.")
        # After cloning, run git lfs pull to download the actual files
        print_rank(f"  Running git lfs pull for {repo_id}...")
        lfs_command = f"cd {output_path} && git lfs pull"
        lfs_result = os.system(lfs_command)
        if lfs_result != 0:
            print_rank(f"  Failed to pull Git LFS files for {repo_id}. Exit code: {lfs_result}")
        else:
            print_rank(f"  Successfully pulled Git LFS files for {repo_id}.")


def download_datasets_via_git_clone(datasets_map: dict, base_output_dir: str, modality_name: str):
    """
    Downloads datasets by git cloning their Hugging Face repositories.

    Args:
        datasets_map (dict): A dictionary of dataset names to (repo_id, subset, split) tuples.
        base_output_dir (str): The base directory to save the cloned repositories.
        modality_name (str): The name of the modality (e.g., "visdoc-tasks", "video-tasks").
    """
    print(f"\n" + "="*20 + f" Cloning {modality_name.replace('-', ' ').title()} Datasets " + "="*20)
    
    for dataset_name, (repo_id, _, _) in datasets_map.items():
        # Construct the full output path for this dataset
        # The repo_id might contain slashes, so we need to make sure the last part is the dataset folder name
        # e.g., "vidore/esg_reports_human_labeled_v2" -> "esg_reports_human_labeled_v2"
        dataset_folder_name = repo_id.split('/')[-1]
        output_path = os.path.join(base_output_dir, dataset_folder_name)
        
        clone_hf_dataset(repo_id, output_path)


if __name__ == "__main__":
    # Download all dataset types by calling the respective functions
    # Note: this doesn't work so well especially for visdoc tasks as often there are multiple subsets, Use clone_hf_dataset instead
    # download_visdoc_datasets(VISDOC_DATASETS, output_dir="hf_datasets/visdoc-tasks")
    # download_standard_datasets(VIDEO_DATASETS, "video", output_dir="hf_datasets/video-tasks")
    # download_standard_datasets(IMAGE_DATASETS, "image", output_dir="hf_datasets/image-tasks")

    # BASE_RAW_DATA_DIR = "/mnt/disks/rmeng_pd/data/vlm2vec/raw"
    BASE_RAW_DATA_DIR = "~/Downloads/vlm2vec/data/"

    # Ensure git lfs is installed and configured
    print_rank("  Ensuring git lfs is installed...")
    os.system("git lfs install")

    VISDOC_OUTPUT_DIR = os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks")
    # VIDEO_OUTPUT_DIR = os.path.join(BASE_RAW_DATA_DIR, "video-tasks")
    # IMAGE_OUTPUT_DIR = os.path.join(BASE_RAW_DATA_DIR, "image-tasks")

    # Ensure the base output directories exist
    os.makedirs(VISDOC_OUTPUT_DIR, exist_ok=True)
    # os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    # os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    # Call the new download function for each modality
    download_datasets_via_git_clone(VISDOC_DATASETS, VISDOC_OUTPUT_DIR, "visdoc-tasks")
    # download_datasets_via_git_clone(VIDEO_DATASETS, VIDEO_OUTPUT_DIR, "video-tasks")
    # download_datasets_via_git_clone(IMAGE_DATASETS, IMAGE_OUTPUT_DIR, "image-tasks")

    print("\nAll dataset cloning attempts are complete.")