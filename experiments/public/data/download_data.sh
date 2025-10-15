#!/usr/bin/env bash

# You may need to install git-lfs to download data properly

set -ex

### Training data

# 1. Change to desired directory for storing data
DATA_BASEDIR=data/vlm2vec_train
mkdir -p "$DATA_BASEDIR"
pushd "$DATA_BASEDIR"

# 2.1 Download image data and unzip
git clone https://huggingface.co/datasets/TIGER-Lab/MMEB-train
pushd MMEB-train
python unzip_file.py
popd

# 2.2 Download LLaVA-Hound video data
sudo apt-get install -y git-lfs
git clone https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/ video
pushd video/

pushd train_300k/
for f in *.tar.gz; do tar -xzvf "$f"; done
popd

pushd train_600k/
for f in *.tar.gz; do tar -xzvf "$f"; done
popd

popd

# 3. Adjust the data path in the data config yaml, refer to `experiments/public/train/train_alltasks.yaml`

popd

### Eval data

DATA_BASEDIR=data/vlm2vec_eval
mkdir -p "$DATA_BASEDIR"
pushd "$DATA_BASEDIR"

git clone https://huggingface.co/datasets/TIGER-Lab/MMEB-V2 .
pushd image-tasks/
tar -xzvf mmeb_v1.tar.gz
mv MMEB/* .
rmdir MMEB
popd

pushd ../video-tasks/frames/
for f in *.tar.gz; do tar -xzvf "$f" -C "${f%%.*}"; done
popd

popd
