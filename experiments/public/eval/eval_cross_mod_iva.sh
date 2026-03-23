DATA_BASEDIR="/home/mingyi/synthesis/mscoco_omini"
OUTPUT_PATH="/home/mingyi/AI-Projects/VLM2VEC_fork/VLM2Vec/eval_cross_modality_outputs/i2tvanemotrontestGlobal"

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --master_port=2277 --max_restarts=0 eval.py \
CUDA_VISIBLE_DEVICES="1" python eval.py \
      --pooling "mean" \
      --normalize true \
      --per_device_eval_batch_size 16 \
      --model_backbone "nvomniembed" \
      --model_name "nvidia/omni-embed-nemotron-3b" \
      --dataset_config "/home/mingyi/AI-Projects/VLM2VEC_fork/VLM2Vec/experiments/public/eval/cross_modality.yaml" \
      --encode_output_path "$OUTPUT_PATH" \
      --data_basedir "$DATA_BASEDIR"