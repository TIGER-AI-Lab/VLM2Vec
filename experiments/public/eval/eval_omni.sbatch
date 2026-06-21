#!/usr/bin/env bash
#SBATCH --job-name=omni
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/rmeng_data/exps/vlm2vec/omni/logs/%x-%j.log

# OmniSET (MSCOCO cross-modal cmret) no-audio eval, run from the olm2vec worktree.
# Usage: sbatch --job-name=omni-<tag> eval_omni.sbatch <model_dir> <tag> <backbone> <pooling> <python>
set -e
MODEL=${1:?model dir}
TAG=${2:?tag}                 # nemotron | q3vl2B | q3vl8B  (output subdir)
BACKBONE=${3:?backbone}       # nvomniembed | qwen3_vl
POOLING=${4:?pooling}         # mean | last
PY=${5:?python}               # /rmeng_data/envs/{vlm2vec,qwen3vl}/bin/python
ENVBIN="$(dirname "$PY")"                    # env bin (ffmpeg/ffprobe live here)
ENVLIB="$(dirname "$ENVBIN")/lib"            # FFmpeg libs for video frame extraction

REPO=/rmeng_data/projects/embed/VLM2Vec-omni      # olm2vec worktree (native omni code)
BASEDIR=/rmeng_data/data/vlm2vec/MMEB-V3/mscoco_omni
CONFIG=experiments/public/eval/cross_modality_no_audio.yaml
OUT=/rmeng_data/exps/vlm2vec/omni/$TAG/
mkdir -p "$OUT" /rmeng_data/exps/vlm2vec/omni/logs

cd "$REPO"
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=$ENVLIB:$LD_LIBRARY_PATH
export PATH=$ENVBIN:$PATH                     # so save_frames finds ffmpeg/ffprobe
export MSCOCO_OMNI_ROTARY_DTYPE_PATCH=1   # authors' rotary-dtype patch for mscoco-omni

echo "==> GPU preflight on $(hostname)"
MINFREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ -z "$MINFREE" ] || [ "$MINFREE" -lt 70000 ] && { echo "ABORT: squatted node (minfree=$MINFREE)"; exit 1; } || true
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()>=1 else 3)" || { echo "ABORT: CUDA/FabricManager not ready"; exit 1; }

echo "==> $(date) omni eval tag=$TAG backbone=$BACKBONE pooling=$POOLING model=$MODEL"
CUDA_VISIBLE_DEVICES=0 $PY eval.py \
  --pooling "$POOLING" --normalize true \
  --per_device_eval_batch_size 16 \
  --model_backbone "$BACKBONE" \
  --model_name "$MODEL" \
  --processor_name "$MODEL" \
  --dataset_config "$CONFIG" \
  --encode_output_path "$OUT" \
  --data_basedir "$BASEDIR"
echo "==> $(date) DONE $TAG"
