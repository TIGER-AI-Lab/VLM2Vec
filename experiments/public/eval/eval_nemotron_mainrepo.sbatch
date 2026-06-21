#!/usr/bin/env bash
#SBATCH --job-name=nemoM
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=/rmeng_data/exps/vlm2vec/nemotron-main/logs/%x-%j.log

# Run MMEB-V3 eval using the MAIN REPO's live loaders (team's local code), not olm2vec branch.
# Usage: sbatch --job-name=nemoM-image eval_nemotron_mainrepo.sbatch image
set -e
MODALITY=${1:?modality}
BASEDIR=${2:-/rmeng_data/data/vlm2vec/MMEB-V3-eval}
CONFIG=${3:-experiments/public/eval/$MODALITY.yaml}   # optional config override
BATCH=${4:-32}
REPO=/rmeng_data/projects/embed/VLM2Vec          # MAIN repo (team's live eval code)
MODEL=/rmeng_data/data/vlm2vec/models/omni-embed-nemotron-3b
PY=/rmeng_data/envs/vlm2vec/bin/python
OUT=/rmeng_data/exps/vlm2vec/nemotron-main/$MODALITY/
mkdir -p "$OUT" /rmeng_data/exps/vlm2vec/nemotron-main/logs

cd "$REPO"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/rmeng_data/envs/vlm2vec/lib:$LD_LIBRARY_PATH   # FFmpeg libs so torchcodec loads

echo "==> GPU preflight on $(hostname)"
nvidia-smi --query-gpu=memory.free --format=csv,noheader || { echo "no GPUs"; exit 1; }
MINFREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ -z "$MINFREE" ] || [ "$MINFREE" -lt 70000 ] && { echo "ABORT: squatted node (minfree=$MINFREE)"; exit 1; } || true
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()>=8 else 3)" || { echo "ABORT: FabricManager/CUDA not ready"; exit 1; }

echo "==> $(date) main-repo eval modality=$MODALITY"
$PY -m torch.distributed.run --nproc_per_node 8 --master_port $((20000 + RANDOM % 10000)) eval.py \
  --pooling mean --normalize true \
  --per_device_eval_batch_size $BATCH \
  --model_backbone nvomniembed \
  --model_name "$MODEL" \
  --processor_name "$MODEL" \
  --dataset_config "$CONFIG" \
  --encode_output_path "$OUT" \
  --data_basedir "$BASEDIR"
echo "==> $(date) DONE $MODALITY"
