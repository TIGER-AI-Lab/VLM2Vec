#!/usr/bin/env bash

echo "==> Environment"
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

# cd projects/VLM2Vec/ || exit

# ==============================================================================
# Configuration
# ==============================================================================
CUDA_VISIBLE_DEVICES="5"
BATCH_SIZE=8
# NPROC=2 多卡
# NPROC=2 
# BATCH_SIZE=8
# MODALITIES=("image" "video" "tool" "visdoc" "audio" "text")
MODALITIES=("gui")
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
# Audio eval option: cap waveform length in seconds (used for MODALITY=audio only)
AUDIO_MAX_SECONDS="${AUDIO_MAX_SECONDS:-300}"

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

# Parse a MODEL_SPECS line into globals.
# Legacy format keeps working:
#   MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH]
# Extended options (append as key=value):
#   processor_name=...;lora=true|false;checkpoint_path=...;pooling=...;normalize=...;audio_max_seconds=...;extra_args=...
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
      extra_args=*) SPEC_EXTRA_ARGS="${opt#extra_args=}" ;;
      "") ;;
      *) echo "WARNING: Unknown MODEL_SPECS option '$opt' in spec: $spec" ;;
    esac
  done
}

# ==> Define models and their base output paths here
# Format: "MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH]"
declare -a MODEL_SPECS
# 选项1: 使用本地Qwen2-VL-2B + VLM2Vec-V2.0适配器（推荐，避免网络下载）
# MODEL_SPECS+=( "/code/.cache/huggingface/Qwen2-VL-2B;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0-Qwen2VL-2B;/code/.cache/huggingface/VLM2Vec-V2.0" )
# MODEL_SPECS+=( "/code/.cache/huggingface/Qwen2.5-Omni-3B;qwen2_5_omni;$OUTPUT_BASEDIR/VLM2Vec-V3.0-Qwen2_5omni-3B;/code/OLM2VEC_and_MMEB-V3/VLM2Vec1/VLM2Vec/exps/output_model/Qwen2_5Omni_3B.audio.lora16.BS512.IB64.GCq8p8.NormTemp002.lr5e5.step5kwarm100/checkpoint-2850" )

MODEL_SPECS+=( "/data/mengrui/.cache/huggingface/omni-embed-nemotron-3b;nvomniembed;$OUTPUT_BASEDIR/omni-embed-nemotron-3b" )
# Ours example (only edit MODEL_SPECS when switching models):
# MODEL_SPECS+=( "/data/mengrui/.cache/huggingface/Qwen2.5-Omni-3B;qwen2_5_omni;$OUTPUT_BASEDIR/ours;/abs/path/checkpoint-7000;processor_name=/data/mengrui/.cache/huggingface/Qwen2.5-Omni-3B;lora=true" )
# MODEL_SPECS+=( "/data/mengrui/.cache/huggingface/WAVE-7B;wave;$OUTPUT_BASEDIR/WAVE-7B" )
# MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-2B-Instruct" )
# MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct" )
# MODEL_SPECS+=( "code-kunkun/LamRA-Ret;lamra;$OUTPUT_BASEDIR/LamRA-Ret" )
# MODEL_SPECS+=( "vidore/colpali-v1.3;colpali;$OUTPUT_BASEDIR/colpali-v1.3" )


# ==============================================================================
# Main Execution Loop
# ==============================================================================
mkdir -p "$OUTPUT_BASEDIR"
echo "model_name,modality,start_time,end_time,duration_seconds,duration_hms,status" > "$TIMING_LOG"
echo "==> Timing log: $TIMING_LOG"

global_start_ts=$(date +%s)

# Loop through each model specification
for spec in "${MODEL_SPECS[@]}"; do
  # Parse model spec (legacy + extended options)
  parse_model_spec "$spec"
  model_start_ts=$(date +%s)
  EXTRA_MODEL_ARGS=""

  if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_BACKBONE" ] || [ -z "$BASE_OUTPUT_PATH" ]; then
    echo "Invalid MODEL_SPECS entry: $spec"
    echo "Expected: MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH][;key=value...]"
    exit 1
  fi

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

  echo "================================================="
  echo "🚀 Processing Model: $MODEL_NAME"
  echo "================================================="

  # Loop through each modality for the current model
  for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="experiments/public/eval/$MODALITY.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"
    EXTRA_DATA_ARGS=""
    if [[ "$MODALITY" == "audio" ]]; then
      EFFECTIVE_AUDIO_MAX_SECONDS="${SPEC_AUDIO_MAX_SECONDS:-$AUDIO_MAX_SECONDS}"
      EXTRA_DATA_ARGS="$EXTRA_DATA_ARGS --audio_max_seconds $EFFECTIVE_AUDIO_MAX_SECONDS"
    fi

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"

    # Ensure the output directory exists
    mkdir -p "$OUTPUT_PATH"
# --pooling eos
#   cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node $NPROC eval.py \ duoka
    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python eval.py \
      --pooling \"$SPEC_POOLING\" \
      --normalize \"$SPEC_NORMALIZE\" \
      --per_device_eval_batch_size $BATCH_SIZE \
      --model_backbone \"$MODEL_BACKBONE\" \
      --model_name \"$MODEL_NAME\" \
      $EXTRA_MODEL_ARGS \
      $EXTRA_DATA_ARGS \
      --dataset_config \"$DATA_CONFIG_PATH\" \
      --encode_output_path \"$OUTPUT_PATH\" \
      --data_basedir \"$DATA_BASEDIR\" "
    if [ -n "$SPEC_PROCESSOR_NAME" ]; then
      cmd="$cmd --processor_name \"$SPEC_PROCESSOR_NAME\""
    fi
    if [ -n "$SPEC_LORA" ]; then
      cmd="$cmd --lora \"$SPEC_LORA\""
    fi
    if [ -n "$SPEC_EXTRA_ARGS" ]; then
      cmd="$cmd $SPEC_EXTRA_ARGS"
    fi
    # Add checkpoint_path if specified new added --lora true：--lora true \
      # --processor_name /code/.cache/huggingface/Qwen2.5-Omni-3B
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
done

global_end_ts=$(date +%s)
global_duration_seconds=$((global_end_ts - global_start_ts))
global_duration_hms="$(format_duration "$global_duration_seconds")"

echo "✅ All jobs completed."
echo "Total time: $global_duration_hms (${global_duration_seconds}s)"
echo "Timing details saved to: $TIMING_LOG"
