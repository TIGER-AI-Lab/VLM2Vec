#!/usr/bin/env bash
set -ex

### Eval data only (no training, no video) - using HF transfer

DATA_BASEDIR=data/vlm2vec_eval
mkdir -p "$DATA_BASEDIR"
pushd "$DATA_BASEDIR"

# Install hf_transfer for faster downloads (optional)
pip install hf_transfer -q

# Set environment variable to enable hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download MMEB-V2 eval set using huggingface_hub
if [ ! -d ".git" ]; then
    # Using snapshot_download from huggingface_hub
    python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

# Download the dataset
snapshot_download(
    repo_id="TIGER-Lab/MMEB-V2",
    repo_type="dataset",
    local_dir=".",
    local_dir_use_symlinks=False,
    max_workers=8
)
print("Download complete!")
EOF
else
    echo "MMEB-V2 already exists — skip download"
fi

# Extract only IMAGE tasks
pushd image-tasks/
tar -xzvf mmeb_v1.tar.gz
mv MMEB/* .
rmdir MMEB
popd

popd   # leave eval dir