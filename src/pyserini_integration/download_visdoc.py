import argparse
from huggingface_hub import snapshot_download
import os


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Download visdoc-tasks from Hugging Face Hub"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./MMEB-V2",
        help="Local directory to download the files to (default: ./MMEB-V2)",
    )
    args = parser.parse_args()

    # Define the folder you want
    repo_id = "TIGER-Lab/MMEB-V2"
    local_dir = args.local_dir

    print(f"Starting download of visdoc-tasks from {repo_id} to {local_dir}...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="visdoc-tasks/*",
        local_dir=local_dir,
        resume_download=True,  # Will resume if the connection drops
        max_workers=4,  # Parallel downloads for speed
    )

    print(f"Download complete! Files are in {os.path.abspath(local_dir)}")


if __name__ == "__main__":
    main()
