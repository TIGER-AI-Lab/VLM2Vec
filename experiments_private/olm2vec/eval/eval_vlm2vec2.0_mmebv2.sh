#!/usr/bin/env bash

echo "==> Environment"
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

if [ -f "/home/rmeng_google_com/miniconda3/bin/conda" ]; then
    eval "$(/home/rmeng_google_com/miniconda3/bin/conda shell.bash hook)"
elif [ -f "/rmeng_data/envs/miniconda3/bin/conda" ]; then
    eval "$(/rmeng_data/envs/miniconda3/bin/conda shell.bash hook)"
fi
conda activate /rmeng_data/envs/vlm2vec

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

# ==============================================================================
# Configuration
# ==============================================================================
CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
BATCH_SIZE=16
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
MODALITIES=("image" "video" "visdoc")
DATA_BASEDIR=/rmeng_data/data/vlm2vec/MMEB-V3-eval
OUTPUT_BASEDIR=/rmeng_data/exps/vlm2vec/olm2vec/baselines

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

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

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
  SPEC_POOLING=""
  SPEC_NORMALIZE="true"
  SPEC_AUDIO_MAX_SECONDS=""
  SPEC_MODALITIES=""
  SPEC_EXTRA_ARGS=""

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

declare -a MODEL_SPECS
MODEL_SPECS+=( "Qwen/Qwen2-VL-2B-Instruct;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0;VLM2Vec/VLM2Vec-V2.0;lora=true;pooling=last;normalize=true" )

mkdir -p "$OUTPUT_BASEDIR"
echo "model_name,modality,start_time,end_time,duration_seconds,duration_hms,status" > "$TIMING_LOG"
echo "model_name,model_backbone,modality,dataset_name,start_time,end_time,duration_seconds,duration_hms,load_seconds,query_seconds,cand_seconds,score_seconds,do_query,do_cand,status,error" > "$DATASET_TIMING_LOG"

global_start_ts=$(date +%s)
failed_tasks=0

for spec in "${MODEL_SPECS[@]}"; do
  parse_model_spec "$spec"
  if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_BACKBONE" ] || [ -z "$BASE_OUTPUT_PATH" ]; then
    echo "Invalid MODEL_SPECS entry: $spec"
    exit 1
  fi

  EFFECTIVE_NORMALIZE="$SPEC_NORMALIZE"
  EFFECTIVE_POOLING="last"
  if [[ -n "$SPEC_POOLING" ]]; then
    EFFECTIVE_POOLING="$SPEC_POOLING"
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

  for MODALITY in "${MODEL_MODALITIES[@]}"; do
    if [[ -z "$MODALITY" ]]; then
      continue
    fi
    DATA_CONFIG_PATH="experiments/public/eval/$MODALITY.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/eval_results/$MODALITY/"
    
    # --- OOM Handling for Video / VisDoc ---
    CURRENT_NPROC=$NPROC_PER_NODE
    CURRENT_BS=$BATCH_SIZE

    if [[ "$MODALITY" == *visdoc* ]]; then
      echo "⚠️  Reducing nproc_per_node to 1 for $MODALITY to prevent OOM"
      CURRENT_NPROC=1
    fi

    if [[ "$MODALITY" == *video* ]]; then
      echo "🎥 Reducing batch size to 4 for $MODALITY to prevent OOM"
      CURRENT_BS=4
    fi

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"
    echo "  - GPU Count: $CURRENT_NPROC"
    echo "  - Batch Size: $CURRENT_BS"

    mkdir -p "$OUTPUT_PATH"

    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES EVAL_MODALITY=\"$MODALITY\" EVAL_DATASET_TIMING_LOG=\"$DATASET_TIMING_LOG\" /rmeng_data/envs/vlm2vec/bin/torchrun --nproc_per_node=$CURRENT_NPROC --master_port=2277 --max_restarts=0 eval.py \
      --pooling \"$EFFECTIVE_POOLING\" \
      --normalize \"$EFFECTIVE_NORMALIZE\" \
      --per_device_eval_batch_size $CURRENT_BS \
      --dataloader_num_workers $DATALOADER_NUM_WORKERS \
      --model_backbone \"$MODEL_BACKBONE\" \
      --model_name \"$MODEL_NAME\" \
      --dataset_config \"$DATA_CONFIG_PATH\" \
      --encode_output_path \"$OUTPUT_PATH\" \
      --data_basedir \"$DATA_BASEDIR\""
    
    if [ -n "$SPEC_PROCESSOR_NAME" ]; then
      cmd="$cmd --processor_name \"$SPEC_PROCESSOR_NAME\""
    fi
    if [ -n "$SPEC_LORA" ]; then
      cmd="$cmd --lora \"$SPEC_LORA\""
    fi
    if [ -n "$CHECKPOINT_PATH" ]; then
      cmd="$cmd --checkpoint_path \"$CHECKPOINT_PATH\""
    fi

    echo "  - Executing command..."
    start_time_human="$(date '+%Y-%m-%d %H:%M:%S')"
    start_ts=$(date +%s)
    
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
