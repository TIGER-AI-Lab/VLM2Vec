#!/usr/bin/env bash
#SBATCH --job-name=q3vl
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=/rmeng_data/exps/vlm2vec/qwen3vl/logs/%x-%j.log

# Qwen3-VL-Embedding eval (MAIN repo loaders, qwen3vl env w/ transformers 4.57.1).
# Usage: sbatch --job-name=q3vl2B-image eval_qwen3vl.sbatch <model_dir> <tag> <modality> [config] [batch]
set -e
MODEL=${1:?model dir}      # /rmeng_data/data/vlm2vec/models/Qwen3-VL-Embedding-2B|8B
TAG=${2:?tag}              # 2B | 8B  (output subdir)
MODALITY=${3:?modality}
CONFIG=${4:-experiments/public/eval/$MODALITY.yaml}
BATCH=${5:-8}
REPO=/rmeng_data/projects/embed/VLM2Vec
PY=/rmeng_data/envs/qwen3vl/bin/python
BASEDIR=/rmeng_data/data/vlm2vec/MMEB-V3-eval
OUT=/rmeng_data/exps/vlm2vec/qwen3vl/$TAG/$MODALITY/
mkdir -p "$OUT" /rmeng_data/exps/vlm2vec/qwen3vl/logs

cd "$REPO"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/rmeng_data/envs/qwen3vl/lib:$LD_LIBRARY_PATH

echo "==> GPU preflight on $(hostname)"
MINFREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ -z "$MINFREE" ] || [ "$MINFREE" -lt 70000 ] && { echo "ABORT: squatted node (minfree=$MINFREE)"; exit 1; } || true
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()>=8 else 3)" || { echo "ABORT: CUDA/FabricManager not ready"; exit 1; }

echo "==> $(date) qwen3vl eval tag=$TAG modality=$MODALITY model=$MODEL"
$PY -m torch.distributed.run --nproc_per_node 8 --master_port $((20000 + RANDOM % 10000)) eval.py \
  --pooling last --normalize true \
  --per_device_eval_batch_size $BATCH \
  --model_backbone qwen3_vl \
  --model_name "$MODEL" \
  --processor_name "$MODEL" \
  --dataset_config "$CONFIG" \
  --encode_output_path "$OUT" \
  --data_basedir "$BASEDIR"
echo "==> $(date) DONE $TAG/$MODALITY"
