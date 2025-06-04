export PYTHONPATH=/home/ziyan/VLM2Vec:$PYTHONPATH  # Change to your own path
export HF_HOME=~/.cache/huggingface

CUDA_VISIBLE_DEVICES=7 python evaluation/eval_flickr.py \
  --model_name TIGER-Lab/VLM2Vec-LLaVa-Next \
  --model_backbone llava_next \
  --pooling last --normalize True \
  --per_device_eval_batch_size 8 \
  --encode_output_path /home/ziyan/MMEB_eval/flickr/


## I -> T:
#Recall@1: 0.9380
#Recall@5: 0.9940
#Recall@10: 0.9960
#
## T -> I
#Recall@1: 0.8022
#Recall@5: 0.9496
#Recall@10: 0.9736
