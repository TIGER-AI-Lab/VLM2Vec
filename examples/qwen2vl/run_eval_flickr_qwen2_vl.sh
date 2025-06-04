export PYTHONPATH=/home/ziyan/VLM2Vec:$PYTHONPATH  # Change to your own path
export HF_HOME=~/.cache/huggingface

CUDA_VISIBLE_DEVICES=7 python evaluation/eval_flickr.py \
  --model_name Qwen/Qwen2-VL-7B-Instruct \
  --checkpoint_path TIGER-Lab/VLM2Vec-Qwen2VL-7B \
  --model_backbone qwen2_vl \
  --pooling last --normalize True \
  --lora True \
  --per_device_eval_batch_size 8 \
  --encode_output_path /home/ziyan/MMEB_eval/flickr/


## I -> T:
#Recall@1: 0.9370
#Recall@5: 0.9960
#Recall@10: 0.9990

## T -> I
#Recall@1: 0.8002
#Recall@5: 0.9530
#Recall@10: 0.9762
