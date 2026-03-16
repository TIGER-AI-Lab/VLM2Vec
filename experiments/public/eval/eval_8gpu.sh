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
CUDA_VISIBLE_DEVICES="0,1,2,3"
BATCH_SIZE=8
# MODALITIES=("image" "video" "tool" "visdoc" "text")
MODALITIES=("image")
DATA_BASEDIR=/data/mengrui/.cache/huggingface/datasets/MMEB-V3
OUTPUT_BASEDIR=exps/vlm2vec
# WAVE-only optional args (only effective when MODEL_BACKBONE=wave)
WAVE_PROCESSOR_PATH="${WAVE_PROCESSOR_PATH:-}"
WAVE_TRAIN_CLASSIFY="${WAVE_TRAIN_CLASSIFY:-true}"
WAVE_CLASSIFY_TYPE="${WAVE_CLASSIFY_TYPE:-all_layer}"
WAVE_PRED_EMBEDS="${WAVE_PRED_EMBEDS:-true}"
WAVE_USE_BEATS="${WAVE_USE_BEATS:-false}"
WAVE_BEATS_PATH="${WAVE_BEATS_PATH:-}"
WAVE_BEATS_ONLY="${WAVE_BEATS_ONLY:-false}"
# Audio eval option: keep empty by default to preserve previous behavior.
AUDIO_MAX_SECONDS="${AUDIO_MAX_SECONDS:-}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TIMING_LOG="$OUTPUT_BASEDIR/eval_timing_${TIMESTAMP}.csv"
DATASET_TIMING_LOG="$OUTPUT_BASEDIR/eval_dataset_timing_${TIMESTAMP}.csv"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES// /}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE="${#GPU_IDS[@]}"
if [ "$NPROC_PER_NODE" -lt 1 ]; then
  echo "Invalid CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
  exit 1
fi

# Format seconds as HH:MM:SS
format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

# Parse a MODEL_SPECS line into globals.
# Legacy format keeps working:
#   MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH]
# Extended options (append as key=value):
#   processor_name=...;lora=true|false;checkpoint_path=...;pooling=...;normalize=...;audio_max_seconds=...;modalities=...;extra_args=...
parse_model_spec() {
  local spec="$1"
  local start_opt_idx=4
  IFS=';' read -r -a SPEC_FIELDS <<< "$spec"

  MODEL_NAME="${SPEC_FIELDS[0]:-}"
  MODEL_BACKBONE="${SPEC_FIELDS[1]:-}"
  BASE_OUTPUT_PATH="${SPEC_FIELDS[2]:-}"
  CHECKPOINT_PATH="${SPEC_FIELDS[3]:-}"

  SPEC_PROCESSOR_NAME=""
  SPEC_LORA=""
  SPEC_POOLING="mean"
  SPEC_NORMALIZE="true"
  SPEC_AUDIO_MAX_SECONDS=""
  SPEC_MODALITIES=""
  SPEC_EXTRA_ARGS=""

  # If the 4th field is key=value, treat it as an option (no legacy checkpoint field).
  if [[ -n "${SPEC_FIELDS[3]:-}" && "${SPEC_FIELDS[3]}" == *=* ]]; then
    CHECKPOINT_PATH=""
    start_opt_idx=3
  fi

  for ((i=start_opt_idx; i<${#SPEC_FIELDS[@]}; i++)); do
    opt="${SPEC_FIELDS[$i]}"
    case "$opt" in
      processor_name=*) SPEC_PROCESSOR_NAME="${opt#processor_name=}" ;;
      lora=*) SPEC_LORA="${opt#lora=}" ;;
      checkpoint_path=*) CHECKPOINT_PATH="${opt#checkpoint_path=}" ;;
      pooling=*) SPEC_POOLING="${opt#pooling=}" ;;
      normalize=*) SPEC_NORMALIZE="${opt#normalize=}" ;;
      audio_max_seconds=*) SPEC_AUDIO_MAX_SECONDS="${opt#audio_max_seconds=}" ;;
      modalities=*) SPEC_MODALITIES="${opt#modalities=}" ;;
      extra_args=*) SPEC_EXTRA_ARGS="${opt#extra_args=}" ;;
      "") ;;
      *) echo "WARNING: Unknown MODEL_SPECS option '$opt' in spec: $spec" ;;
    esac
  done
}


# ==> Define models and their base output paths here
# Format: "MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH]"
declare -a MODEL_SPECS
# MODEL_SPECS+=( "/data/mengrui/.cache/huggingface/omni-embed-nemotron-3b;nvomniembed;$OUTPUT_BASEDIR/omni-embed-nemotron-3b" )
# Ours example (only edit MODEL_SPECS when switching models):
MODEL_SPECS+=( "/data/mengrui/.cache/huggingface/Qwen2.5-Omni-3B;qwen2_5_omni;$OUTPUT_BASEDIR/ours;/data/mengrui/.cache/huggingface/Qwen2_5Omni_3B_BS512_step5k/checkpoint-5000" )
# MODEL_SPECS+=( "/data/mengrui/.cache/huggingface/Qwen3-VL-Embedding-8B;qwen3_vl;$OUTPUT_BASEDIR/Qwen3-VL-Embedding-8B" )
# MODEL_SPECS+=( "/data/mengrui/.cache/huggingface/Qwen3-VL-Embedding-2B;qwen3_vl;$OUTPUT_BASEDIR/Qwen3-VL-Embedding-2B" )
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
echo "model_name,model_backbone,modality,dataset_name,start_time,end_time,duration_seconds,duration_hms,load_seconds,query_seconds,cand_seconds,score_seconds,do_query,do_cand,status,error" > "$DATASET_TIMING_LOG"
echo "==> Timing log: $TIMING_LOG"
echo "==> Dataset timing log: $DATASET_TIMING_LOG"
echo "==> CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==> nproc_per_node: $NPROC_PER_NODE"
echo ""

global_start_ts=$(date +%s)
failed_tasks=0

# Loop through each model specification
for spec in "${MODEL_SPECS[@]}"; do
  # Parse model spec (legacy + extended options)
  parse_model_spec "$spec"
  if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_BACKBONE" ] || [ -z "$BASE_OUTPUT_PATH" ]; then
    echo "Invalid MODEL_SPECS entry: $spec"
    echo "Expected: MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH][;key=value...]"
    exit 1
  fi
  EXTRA_MODEL_ARGS=""
  EFFECTIVE_NORMALIZE="$SPEC_NORMALIZE"
  case "$MODEL_BACKBONE" in
    qwen2_5_omni|nvomniembed|wave)
      EFFECTIVE_NORMALIZE="true"
      ;;
  esac
  if [[ "$MODEL_BACKBONE" == "wave" ]]; then
    WAVE_PROCESSOR_EFFECTIVE="${WAVE_PROCESSOR_PATH:-$MODEL_NAME}"
    EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS --processor_name \"$WAVE_PROCESSOR_EFFECTIVE\""
    EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS --wave_train_classify \"$WAVE_TRAIN_CLASSIFY\""
    EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS --wave_classify_type \"$WAVE_CLASSIFY_TYPE\""
    EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS --wave_pred_embeds \"$WAVE_PRED_EMBEDS\""
    if [[ "$WAVE_USE_BEATS" == "true" ]]; then
      if [[ -n "$WAVE_BEATS_PATH" ]]; then
        EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS --wave_use_beats true --wave_beats_path \"$WAVE_BEATS_PATH\" --wave_beats_only \"$WAVE_BEATS_ONLY\""
      else
        echo "WARNING: WAVE_USE_BEATS=true but WAVE_BEATS_PATH is empty; fallback to --wave_use_beats false."
        EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS --wave_use_beats false"
      fi
    else
      EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS --wave_use_beats false"
    fi
  fi
  MODEL_MODALITIES=("${MODALITIES[@]}")
  if [[ -n "$SPEC_MODALITIES" ]]; then
    SPEC_MODALITIES="${SPEC_MODALITIES// /}"
    IFS=',' read -r -a MODEL_MODALITIES <<< "$SPEC_MODALITIES"
  fi
  model_start_ts=$(date +%s)

  echo "================================================="
  echo "🚀 Processing Model: $MODEL_NAME"
  echo "   Modalities: ${MODEL_MODALITIES[*]}"
  echo "================================================="

  # Loop through each modality for the current model
  for MODALITY in "${MODEL_MODALITIES[@]}"; do
    if [[ -z "$MODALITY" ]]; then
      continue
    fi
    DATA_CONFIG_PATH="experiments/public/eval/$MODALITY.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"
    EXTRA_DATA_ARGS=""
    if [[ "$MODALITY" == "audio" ]]; then
      EFFECTIVE_AUDIO_MAX_SECONDS="${SPEC_AUDIO_MAX_SECONDS:-$AUDIO_MAX_SECONDS}"
      if [[ -n "$EFFECTIVE_AUDIO_MAX_SECONDS" ]]; then
        EXTRA_DATA_ARGS="$EXTRA_DATA_ARGS --audio_max_seconds $EFFECTIVE_AUDIO_MAX_SECONDS"
      fi
    fi
    if [ "$BASE_OUTPUT_PATH" = "/" ] || [ -z "$BASE_OUTPUT_PATH" ]; then
      echo "Invalid BASE_OUTPUT_PATH resolved to '$BASE_OUTPUT_PATH' for spec: $spec"
      exit 1
    fi

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"

    # Ensure the output directory exists
    mkdir -p "$OUTPUT_PATH"

    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES EVAL_MODALITY=\"$MODALITY\" EVAL_DATASET_TIMING_LOG=\"$DATASET_TIMING_LOG\" torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=2277 --max_restarts=0 eval.py \
      --pooling \"mean\" \
      --normalize \"$EFFECTIVE_NORMALIZE\" \
      --per_device_eval_batch_size $BATCH_SIZE \
      --model_backbone \"$MODEL_BACKBONE\" \
      --model_name \"$MODEL_NAME\" \
      $EXTRA_MODEL_ARGS \
      $EXTRA_DATA_ARGS \
      --dataset_config \"$DATA_CONFIG_PATH\" \
      --encode_output_path \"$OUTPUT_PATH\" \
      --lora true \
      --data_basedir \"$DATA_BASEDIR\""
    if [ -n "$SPEC_PROCESSOR_NAME" ]; then
      cmd="$cmd --processor_name \"$SPEC_PROCESSOR_NAME\""
    fi
    if [ -n "$SPEC_LORA" ]; then
      cmd="$cmd --lora \"$SPEC_LORA\""
    fi
    if [ -n "$SPEC_EXTRA_ARGS" ]; then
      cmd="$cmd $SPEC_EXTRA_ARGS"
    fi

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
