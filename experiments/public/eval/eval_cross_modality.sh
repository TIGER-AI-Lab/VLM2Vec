DATA_BASEDIR="/data/mingyi/MMEB/image/MMEB"
OUTPUT_PATH="/home/mingyi/AI-Projects/VLM2VEC_fork/VLM2Vec/eval_cross_modality_outputs/MixedCandFixedQwen2VL2b"

CUDA_VISIBLE_DEVICES="1" python eval.py \
      --pooling eos \
      --normalize true \
      --per_device_eval_batch_size 8 \
      --model_backbone qwen2_vl \
      --model_name Qwen/Qwen2-VL-2b-Instruct \
      --dataset_config "/home/mingyi/AI-Projects/VLM2VEC_fork/VLM2Vec/experiments/public/eval/cross_modality.yaml" \
      --encode_output_path "$OUTPUT_PATH" \
      --data_basedir "$DATA_BASEDIR"