#!/usr/bin/env bash
set -e

### Training data

DATA_BASEDIR=data/vlm2vec_train
mkdir -p "$DATA_BASEDIR"
pushd "$DATA_BASEDIR"

# 2.1 Download MMEB-train with retry and resume
echo "Downloading MMEB-train with retry logic..."
python3 << 'EOF'
import os
import time
import sys
from huggingface_hub import snapshot_download

TARGET_FILES = 84
MAX_RETRIES = 50  # Increased for network issues
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def count_files(directory):
    """Count non-git files in directory"""
    count = 0
    for root, dirs, files in os.walk(directory):
        if '.git' in root:
            continue
        count += len(files)
    return count

for attempt in range(MAX_RETRIES):
    print(f"\n{'='*60}")
    print(f"Attempt {attempt + 1}/{MAX_RETRIES}: Downloading MMEB-train...")
    print(f"{'='*60}")
    
    try:
        snapshot_download(
            repo_id="TIGER-Lab/MMEB-train",
            repo_type="dataset",
            local_dir="MMEB-train",
            local_dir_use_symlinks=False,
            endpoint="https://hf-mirror.com",
            resume_download=True,  # Critical: resume from where it left off
            max_workers=4,  # Limit concurrent downloads
        )
        
        # Count files
        file_count = count_files("MMEB-train")
        print(f"\n✓ Found {file_count} files (target: {TARGET_FILES})")
        
        if file_count >= TARGET_FILES:
            print("✓ Successfully downloaded all files!")
            sys.exit(0)
        else:
            print(f"⚠ Incomplete: {file_count}/{TARGET_FILES} files")
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        # Check if it's a timeout or connection error
        if any(x in error_msg.lower() for x in ['timeout', 'connection', 'read']):
            print("Network error detected, will retry...")
        else:
            print(f"Unexpected error: {e}")
    
    if attempt < MAX_RETRIES - 1:
        wait_time = min(30, (attempt + 1) * 5)  # Exponential backoff, max 30s
        print(f"Waiting {wait_time} seconds before retry...")
        time.sleep(wait_time)
else:
    print(f"\n❌ Failed to download after {MAX_RETRIES} attempts")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "Running unzip_file.py..."
    pushd MMEB-train
    python unzip_file.py
    popd
else
    echo "❌ Download failed"
    exit 1
fi

# 2.2 Download LLaVA-Hound video data (with retry)
echo "Downloading video data..."
for i in {1..10}; do
    if git clone https://hf-mirror.com/datasets/ShareGPTVideo/train_video_and_instruction/ video 2>/dev/null; then
        break
    fi
    echo "Video clone failed, attempt $i/10, retrying..."
    rm -rf video
    sleep 10
done

pushd video/

pushd train_300k/
for f in *.tar.gz; do tar -xzvf "$f"; done
popd

pushd train_600k/
for f in *.tar.gz; do tar -xzvf "$f"; done
popd

popd

popd

### Eval data (with retry)

DATA_BASEDIR=data/vlm2vec_eval
mkdir -p "$DATA_BASEDIR"
pushd "$DATA_BASEDIR"

for i in {1..10}; do
    if git clone https://hf-mirror.com/datasets/TIGER-Lab/MMEB-V2 . 2>/dev/null; then
        break
    fi
    echo "MMEB-V2 clone failed, attempt $i/10, retrying..."
    rm -rf .git
    sleep 10
done

pushd image-tasks/
tar -xzvf mmeb_v1.tar.gz
mv MMEB/* .
rmdir MMEB
popd

pushd video-tasks/frames/
mkdir -p video_cls video_ret
tar -xzvf video_cls.tar.gz -C video_cls
tar -xzvf video_mret.tar.gz
tar -xzvf video_ret.tar.gz -C video_ret
cat video_qa.tar.gz-0{0..4} | tar -xzv
popd

popd

echo "✓ All data downloaded successfully!"