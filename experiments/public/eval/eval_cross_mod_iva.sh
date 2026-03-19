DATA_BASEDIR="/home/mingyi/synthesis/mscoco_omini"
OUTPUT_PATH="/home/mingyi/AI-Projects/VLM2VEC_fork/VLM2Vec/eval_cross_modality_outputs/qwen2.5omni_t2iva_test"

CUDA_VISIBLE_DEVICES="1" python eval.py \
      --pooling mean \
      --normalize true \
      --per_device_eval_batch_size 32 \
      --model_backbone qwen2_5_omni \
      --model_name Qwen/Qwen2.5-Omni-3B \
      --dataset_config "/home/mingyi/AI-Projects/VLM2VEC_fork/VLM2Vec/experiments/public/eval/cross_modality.yaml" \
      --encode_output_path "$OUTPUT_PATH" \
      --data_basedir "$DATA_BASEDIR"