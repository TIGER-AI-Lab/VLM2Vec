#!/usr/bin/env bash

echo "==> Environment"
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# ==============================================================================
# Configuration
# ==============================================================================
CUDA_VISIBLE_DEVICES="4,5,6,7"
BATCH_SIZE=8
MODALITIES=("image" "video" "tool" "visdoc" "audio" "text")
DATA_BASEDIR=/data/mengrui/.cache/huggingface/datasets/MMEB-V3
OUTPUT_BASEDIR=exps/vlm2vec
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TIMING_LOG="$OUTPUT_BASEDIR/eval_timing_${TIMESTAMP}.csv"

# Format seconds as HH:MM:SS
format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}


# ==> Define models and their base output paths here
# Format: "MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH]"
declare -a MODEL_SPECS
MODEL_SPECS+=( "/data/mengrui/.cache/huggingface/omni-embed-nemotron-3b;nvomniembed;$OUTPUT_BASEDIR/omni-embed-nemotron-3b" )
# MODEL_SPECS+=( "VLM2Vec/VLM2Vec-V2.0;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0-Qwen2VL-2B" )
# MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-2B-Instruct" )
# MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct" )
# MODEL_SPECS+=( "code-kunkun/LamRA-Ret;lamra;$OUTPUT_BASEDIR/LamRA-Ret" )
# MODEL_SPECS+=( "vidore/colpali-v1.3;colpali;$OUTPUT_BASEDIR/colpali-v1.3" )
#MODEL_SPECS+=( "royokong/e5-v;llava_next;$OUTPUT_BASEDIR/e5-v" )
#MODEL_SPECS+=( "src/model/vlm_backbone/internvideo2/;internvideo2;$OUTPUT_BASEDIR/internvideo2" )
#MODEL_SPECS+=( "code-kunkun/LamRA-Ret-Qwen2.5VL-7b;lamra-qwen25;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct" )  # not ready


# ==============================================================================
# Main Execution Loop
# ==============================================================================
mkdir -p "$OUTPUT_BASEDIR"
echo "model_name,modality,start_time,end_time,duration_seconds,duration_hms,status" > "$TIMING_LOG"
echo "==> Timing log: $TIMING_LOG"
echo ""

global_start_ts=$(date +%s)
failed_tasks=0

# Loop through each model specification
for spec in "${MODEL_SPECS[@]}"; do
  # Parse the model specification from the spec string
  IFS=';' read -r MODEL_NAME MODEL_BACKBONE BASE_OUTPUT_PATH CHECKPOINT_PATH <<< "$spec"
  if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_BACKBONE" ] || [ -z "$BASE_OUTPUT_PATH" ]; then
    echo "Invalid MODEL_SPECS entry: $spec"
    echo "Expected format: MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH]"
    exit 1
  fi
  model_start_ts=$(date +%s)

  echo "================================================="
  echo "🚀 Processing Model: $MODEL_NAME"
  echo "================================================="

  # Loop through each modality for the current model
  for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="experiments/public/eval/$MODALITY.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"
    if [ "$BASE_OUTPUT_PATH" = "/" ] || [ -z "$BASE_OUTPUT_PATH" ]; then
      echo "Invalid BASE_OUTPUT_PATH resolved to '$BASE_OUTPUT_PATH' for spec: $spec"
      exit 1
    fi

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"

    # Ensure the output directory exists
    mkdir -p "$OUTPUT_PATH"

    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=8 --master_port=2277 --max_restarts=0 eval.py \
      --pooling mean \
      --normalize true \
      --per_device_eval_batch_size $BATCH_SIZE \
      --model_backbone \"$MODEL_BACKBONE\" \
      --model_name \"$MODEL_NAME\" \
      --dataset_config \"$DATA_CONFIG_PATH\" \
      --encode_output_path \"$OUTPUT_PATH\" \
      --data_basedir \"$DATA_BASEDIR\""

    # Add checkpoint_path only when provided in MODEL_SPECS.
    if [ -n "$CHECKPOINT_PATH" ]; then
      cmd="$cmd --checkpoint_path \"$CHECKPOINT_PATH\""
    fi

    echo "  - Executing command..."
    start_time_human="$(date '+%Y-%m-%d %H:%M:%S')"
    start_ts=$(date +%s)
    # echo "$cmd" # Uncomment for debugging the exact command
    if eval "$cmd"; then
      status="success"
    else
      status="failed"
      failed_tasks=$((failed_tasks + 1))
    fi
    end_ts=$(date +%s)
    end_time_human="$(date '+%Y-%m-%d %H:%M:%S')"
    duration_seconds=$((end_ts - start_ts))
    duration_hms="$(format_duration "$duration_seconds")"
    echo "  - Done. status=$status, duration=$duration_hms (${duration_seconds}s)"
    echo "$MODEL_NAME,$MODALITY,$start_time_human,$end_time_human,$duration_seconds,$duration_hms,$status" >> "$TIMING_LOG"
    echo "-------------------------------------------------"
  done

  model_end_ts=$(date +%s)
  model_duration_seconds=$((model_end_ts - model_start_ts))
  model_duration_hms="$(format_duration "$model_duration_seconds")"
  echo "Model total time: $model_duration_hms (${model_duration_seconds}s)"
  echo ""
done

global_end_ts=$(date +%s)
global_duration_seconds=$((global_end_ts - global_start_ts))
global_duration_hms="$(format_duration "$global_duration_seconds")"

echo "✅ All jobs completed."
echo "Total time: $global_duration_hms (${global_duration_seconds}s)"
echo "Failed tasks: $failed_tasks"
echo "Timing details saved to: $TIMING_LOG"

if [ "$failed_tasks" -gt 0 ]; then
  exit 1
fi
