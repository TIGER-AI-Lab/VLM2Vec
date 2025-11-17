#!/bin/bash
#SBATCH --job-name=embedding_infer
#SBATCH -A hexm-critical
#SBATCH --partition=critical
#SBATCH --output=/public/home/xiaojw2025/Workspace/RAHP/slurm/logs/embedding_infer_%j.out
#SBATCH --error=/public/home/xiaojw2025/Workspace/RAHP/slurm/logs/embedding_infer_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=all
#SBATCH --mail-user=mxzxx093834@163.com



set -e
source ~/.bashrc 

echo "model /inspurfs/group/hexm/ov_sgg/train_model_param/train_5k_ratio"

echo "🔧 激活conda环境..."
source /public/home/xiaojw2025/anaconda3/etc/profile.d/conda.sh
conda activate qwen || { echo "❌ 无法激活环境"; exit 1; }

# 切换到项目目录
cd /public/home/xiaojw2025/Workspace/VLM2Vec || { echo "❌ 项目目录不存在"; exit 1; }




export FLASH_ATTENTION_DISABLE=0
export USE_FLASH_ATTENTION=1
export FLASH_ATTENTION_FORCE_BUILD=0
export FLASH_ATTENTION_SKIP_CUDA_BUILD=0

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"


python /public/home/xiaojw2025/Workspace/VLM2Vec/embedding/infer/predict_scene_graph_recall.py


