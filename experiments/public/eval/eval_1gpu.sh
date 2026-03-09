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
CUDA_VISIBLE_DEVICES="4"
BATCH_SIZE=16
# NPROC=4 多卡
# NPROC=4 
# BATCH_SIZE=16
# MODALITIES=("image" "video" "tool" "visdoc" "audio" "text")
MODALITIES=("audio" "image")
DATA_BASEDIR=/data/mengrui/.cache/huggingface/datasets/MMEB-V3
OUTPUT_BASEDIR=exps/vlm2vec


# ==> Define models and their base output paths here
# Format: "MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH]"
declare -a MODEL_SPECS
# 选项1: 使用本地Qwen2-VL-2B + VLM2Vec-V2.0适配器（推荐，避免网络下载）
# MODEL_SPECS+=( "/code/.cache/huggingface/Qwen2-VL-2B;qwen2_vl;$OUTPUT_BASEDIR/VLM2Vec-V2.0-Qwen2VL-2B;/code/.cache/huggingface/VLM2Vec-V2.0" )
# MODEL_SPECS+=( "/code/.cache/huggingface/omni-embed-nemotron-3b;nvomniembed;$OUTPUT_BASEDIR/omni-embed-nemotron-3b" )

MODEL_SPECS+=( "/data/mengrui/.cache/huggingface/Qwen2.5-Omni-3B;qwen2_5_omni;$OUTPUT_BASEDIR/ours;/data/mengrui/OLM2Vec/OLM2Vec/exps/output_model/audio-tool-tasks-jepa+infonce-alltasks/checkpoint-2800" )
# MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-2B-Instruct" )
# MODEL_SPECS+=( "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct;gme;$OUTPUT_BASEDIR/gme-Qwen2-VL-7B-Instruct" )
# MODEL_SPECS+=( "code-kunkun/LamRA-Ret;lamra;$OUTPUT_BASEDIR/LamRA-Ret" )
# MODEL_SPECS+=( "vidore/colpali-v1.3;colpali;$OUTPUT_BASEDIR/colpali-v1.3" )


# ==============================================================================
# Main Execution Loop
# ==============================================================================
# Loop through each model specification
for spec in "${MODEL_SPECS[@]}"; do
  # Parse the model specification: MODEL_NAME;MODEL_BACKBONE;BASE_OUTPUT_PATH[;CHECKPOINT_PATH]
  IFS=';' read -r MODEL_NAME MODEL_BACKBONE BASE_OUTPUT_PATH CHECKPOINT_PATH <<< "$spec"

  echo "================================================="
  echo "🚀 Processing Model: $MODEL_NAME"
  echo "================================================="

  # Loop through each modality for the current model
  for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="experiments/public/eval/$MODALITY.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"

    # Ensure the output directory exists
    mkdir -p "$OUTPUT_PATH"
# --pooling eos
    # cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node $NPROC eval.py \
    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python eval.py \
      --pooling mean \
      --normalize true \
      --per_device_eval_batch_size $BATCH_SIZE \
      --model_backbone \"$MODEL_BACKBONE\" \
      --model_name \"$MODEL_NAME\" \
      --processor_name /data/mengrui/.cache/huggingface/Qwen2.5-Omni-3B \
      --dataset_config \"$DATA_CONFIG_PATH\" \
      --encode_output_path \"$OUTPUT_PATH\" \
      --data_basedir \"$DATA_BASEDIR\""
    # Add checkpoint_path if specified new added --lora true：--lora true \
      # --processor_name /code/.cache/huggingface/Qwen2.5-Omni-3B
    if [ -n "$CHECKPOINT_PATH" ]; then
      cmd="$cmd --lora true --checkpoint_path \"$CHECKPOINT_PATH\""
    fi
    
    echo "  - Executing command..."
    # echo "$cmd" # Uncomment for debugging the exact command
    eval "$cmd"
    echo "  - Done."
    echo "-------------------------------------------------"
  done
done

echo "✅ All jobs completed."
