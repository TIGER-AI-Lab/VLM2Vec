#!/usr/bin/env bash
set -euo pipefail

echo "==> Environment"
echo "conda location: $(which conda || true)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------------------------
# WAVE-only evaluation script.
# This script is isolated from eval_1gpu.sh to avoid impacting other model runs.
# ------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MODALITIES=(${MODALITIES:-audio image video text tool visdoc})
DATA_BASEDIR="${DATA_BASEDIR:-/data/mengrui/.cache/huggingface/datasets/MMEB-V3}"
OUTPUT_BASEDIR="${OUTPUT_BASEDIR:-exps/vlm2vec}"

WAVE_MODEL_PATH="${WAVE_MODEL_PATH:-/data/mengrui/.cache/huggingface/WAVE-7B}"
WAVE_PROCESSOR_PATH="${WAVE_PROCESSOR_PATH:-$WAVE_MODEL_PATH}"
WAVE_BEATS_PATH="${WAVE_BEATS_PATH:-}"
WAVE_BEATS_ONLY="${WAVE_BEATS_ONLY:-false}"

BASE_OUTPUT_PATH="$OUTPUT_BASEDIR/WAVE-7B"
mkdir -p "$BASE_OUTPUT_PATH"

if [[ -n "$WAVE_BEATS_PATH" ]]; then
  WAVE_BEATS_ARGS="--wave_use_beats true --wave_beats_path \"$WAVE_BEATS_PATH\" --wave_beats_only \"$WAVE_BEATS_ONLY\""
  echo "Using BEATs checkpoint: $WAVE_BEATS_PATH"
else
  WAVE_BEATS_ARGS="--wave_use_beats false"
  echo "WARNING: WAVE_BEATS_PATH is empty; this is NOT strict WAVE README reproduction."
fi

for MODALITY in "${MODALITIES[@]}"; do
  DATA_CONFIG_PATH="experiments/public/eval/$MODALITY.yaml"
  OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"
  mkdir -p "$OUTPUT_PATH"

  cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python eval.py \
    --pooling mean \
    --normalize true \
    --per_device_eval_batch_size $BATCH_SIZE \
    --model_backbone wave \
    --model_name \"$WAVE_MODEL_PATH\" \
    --processor_name \"$WAVE_PROCESSOR_PATH\" \
    --wave_train_classify true \
    --wave_classify_type all_layer \
    --wave_pred_embeds true \
    $WAVE_BEATS_ARGS \
    --dataset_config \"$DATA_CONFIG_PATH\" \
    --encode_output_path \"$OUTPUT_PATH\" \
    --data_basedir \"$DATA_BASEDIR\""

  echo "-------------------------------------------------"
  echo "Modality: $MODALITY"
  echo "Output: $OUTPUT_PATH"
  eval "$cmd"
done

echo "WAVE-only evaluation completed."
