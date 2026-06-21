#!/usr/bin/env bash
#SBATCH --job-name=nemo-eval
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=logs/nemo-%x-%j.log

# Usage: sbatch --job-name=nemo-tool eval_nemotron.sbatch tool [data_basedir]
#        sbatch --job-name=nemo-text eval_nemotron.sbatch text /rmeng_data/data/vlm2vec/MMEB-V3-eval
set -e
MODALITY=${1:?pass modality: tool | memory | text | visdoc | audio | video | image | gui}
BASEDIR=${2:-/rmeng_data/data/vlm2vec/MMEB-V3-evalroot}

cd /rmeng_data/projects/embed/VLM2Vec-olm2vec
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=/rmeng_data/data/vlm2vec/models/omni-embed-nemotron-3b
PY=/rmeng_data/envs/vlm2vec/bin/python
NPROC=8
BATCH_SIZE=32
OUT=exps/nemotron/$MODALITY/
mkdir -p "$OUT"

# Preflight: bail fast on bad nodes so we don't waste a run.
#  (a) rogue non-SLURM squatter (e.g. root sglang) -> GPU memory not free -> would OOM.
#  (b) Fabric Manager not initialized (CUDA error 802) -> nvidia-smi works but torch sees 0 GPUs.
echo "==> GPU preflight on $(hostname)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader || { echo "no GPUs on node"; exit 1; }
MINFREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
if [ -z "$MINFREE" ] || [ "$MINFREE" -lt 70000 ]; then
  echo "ABORT: a GPU has <70GB free (min free=${MINFREE}MiB) — node likely squatted. Resubmit."
  exit 1
fi
$PY -c "import torch,sys; n=torch.cuda.device_count(); print('torch CUDA devices:',n); sys.exit(0 if n>=$NPROC else 3)" || {
  echo "ABORT: torch sees <$NPROC CUDA devices (likely Fabric Manager not ready, CUDA err 802). Resubmit."
  exit 1
}

echo "==> $(date) modality=$MODALITY nproc=$NPROC bs=$BATCH_SIZE"
$PY -m torch.distributed.run --nproc_per_node $NPROC --master_port $((20000 + RANDOM % 10000)) eval.py \
  --pooling mean --normalize true \
  --per_device_eval_batch_size $BATCH_SIZE \
  --model_backbone nvomniembed \
  --model_name "$MODEL" \
  --processor_name "$MODEL" \
  --dataset_config experiments/public/eval/$MODALITY.yaml \
  --encode_output_path "$OUT" \
  --data_basedir "$BASEDIR"
echo "==> $(date) DONE $MODALITY"
