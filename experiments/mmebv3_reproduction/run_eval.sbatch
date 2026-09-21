#!/usr/bin/env bash
#SBATCH --job-name=mmebv3_eval
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Universal MMEB-V3 eval runner (one script for nemotron / Qwen3-VL / OmniSET / audio).
# Fully env-var driven — no hardcoded machine paths. See REPRODUCTION.md.
#
# Required:
#   MODEL_PATH       local path or HF id of the embedding model
#   MODEL_BACKBONE   nvomniembed | qwen3_vl | e5_omni | lco_omni | wave
#   DATASET_CONFIG   path to a dataset yaml (e.g. experiments/public/eval/audio.yaml)
#   DATA_BASEDIR     root that the config's relative roots resolve against (== MMEB_V3_DATA_DIR)
#   OUTPUT_PATH      where {dataset}_score.json are written (resumable cache)
#   ENV_PYTHON       /path/to/conda-env/bin/python  (env's bin must contain ffmpeg/ffprobe)
# Optional:
#   POOLING=mean(nvomni)|last(qwen3vl)   BATCH=32   NPROC=8
#   AUDIO_MAX_SECONDS=30   (REQUIRED for audio: avoids the 10.24s _crop_audio default)
#   REPO=<this checkout>   (defaults to $SLURM_SUBMIT_DIR or cwd)
#   RESIZE_MIN_PIXELS / RESIZE_MAX_PIXELS   per-image pixel budget. Defaults are the
#       Qwen2-VL-era 28*28*4 / 28*28*1280. Qwen3-VL-Embedding's own reference impl uses
#       4*32*32=4096 / 1800*32*32=1843200 — pass those to match the published numbers.
#   EXTRA_PYTHONPATH   prepended to PYTHONPATH, e.g. a side-installed qwen-vl-utils>=0.0.14
#       (Qwen3-VL needs image_patch_size=16; 0.0.8 silently resizes on the 28-px grid).
#   VIDEO_MAX_FRAMES / VIDEO_FRAME_SIZE   candidate-side video frames. The eval collator caps
#       candidate videos at --video_max_frames (default 8) and squares each frame to
#       --video_frame_size px (default 224). A 64-frame config needs VIDEO_MAX_FRAMES=64
#       VIDEO_FRAME_SIZE=0 (0 = keep native frame size) or the retrieval sets run at 8x224.
# Qwen3-VL-Embedding on MMEB-v1/v2 as the leaderboard scores it: configs/mmeb_v1_image.yaml,
# configs/mmeb_v2_visdoc.yaml (24 sets, *-page-fixed), configs/mmeb_v2_video_64f.yaml, with
# the RESIZE_* values above. See docs/mmebv2_reproduction/README.md.
set -euo pipefail
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
MODEL_BACKBONE=${MODEL_BACKBONE:?set MODEL_BACKBONE}
DATASET_CONFIG=${DATASET_CONFIG:?set DATASET_CONFIG}
DATA_BASEDIR=${DATA_BASEDIR:?set DATA_BASEDIR}
OUTPUT_PATH=${OUTPUT_PATH:?set OUTPUT_PATH}
ENV_PYTHON=${ENV_PYTHON:?set ENV_PYTHON}
POOLING=${POOLING:-mean}
BATCH=${BATCH:-32}
NPROC=${NPROC:-8}
AUDIO_MAX_SECONDS=${AUDIO_MAX_SECONDS:-}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}

ENVBIN="$(dirname "$ENV_PYTHON")"
ENVLIB="$(dirname "$ENVBIN")/lib"
cd "$REPO"
export PYTHONPATH="${EXTRA_PYTHONPATH:+$EXTRA_PYTHONPATH:}."   # e.g. a side-installed qwen-vl-utils>=0.0.14
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$ENVLIB:${LD_LIBRARY_PATH:-}"   # FFmpeg-7 libs for soundfile/video
export PATH="$ENVBIN:$PATH"                              # ffmpeg/ffprobe for frame/clip extraction
export MMEB_V3_DATA_DIR="$DATA_BASEDIR"
export MSCOCO_OMNI_ROTARY_DTYPE_PATCH=1                  # OmniSET rotary-dtype patch (harmless elsewhere)
mkdir -p "$OUTPUT_PATH"

echo "==> $(date) host=$(hostname) backbone=$MODEL_BACKBONE pooling=$POOLING nproc=$NPROC config=$DATASET_CONFIG"
# GPU preflight: skip squatted / FabricManager-down nodes (resubmit if this aborts)
MINFREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
[ -z "$MINFREE" ] || [ "$MINFREE" -lt 70000 ] && { echo "ABORT: low free mem ($MINFREE MiB) — squatted node"; exit 1; } || true
"$ENV_PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()>=$NPROC else 3)" \
  || { echo "ABORT: <$NPROC visible GPUs (FabricManager/CUDA not ready)"; exit 1; }

EXTRA=()
[ -n "$AUDIO_MAX_SECONDS" ] && EXTRA+=(--audio_max_seconds "$AUDIO_MAX_SECONDS")
[ -n "${RESIZE_MIN_PIXELS:-}" ] && EXTRA+=(--resize_min_pixels "$RESIZE_MIN_PIXELS")
[ -n "${RESIZE_MAX_PIXELS:-}" ] && EXTRA+=(--resize_max_pixels "$RESIZE_MAX_PIXELS")
[ -n "${VIDEO_MAX_FRAMES:-}" ] && EXTRA+=(--video_max_frames "$VIDEO_MAX_FRAMES")
[ -n "${VIDEO_FRAME_SIZE:-}" ] && EXTRA+=(--video_frame_size "$VIDEO_FRAME_SIZE")

"$ENV_PYTHON" -m torch.distributed.run --nproc_per_node "$NPROC" \
  --master_port $((20000 + RANDOM % 10000)) eval.py \
  --pooling "$POOLING" --normalize true \
  --per_device_eval_batch_size "$BATCH" \
  --model_backbone "$MODEL_BACKBONE" \
  --model_name "$MODEL_PATH" --processor_name "$MODEL_PATH" \
  --dataset_config "$DATASET_CONFIG" \
  --encode_output_path "$OUTPUT_PATH" \
  --data_basedir "$DATA_BASEDIR" \
  "${EXTRA[@]}"
echo "==> $(date) DONE -> $OUTPUT_PATH"
