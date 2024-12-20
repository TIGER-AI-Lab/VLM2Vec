export PYTHONPATH=../VLM2Vec/:$PYTHONPATH


CUDA_VISIBLE_DEVICES=1 python evaluation/eval_flickr.py \
  --model_name TIGER-Lab/VLM2Vec-LLaVa-Next \
  --model_backbone llava_next \
  --max_len 256 \
  --pooling last --normalize True \
  --per_device_eval_batch_size 16 \
  --encode_output_path /home/ziyan/MMEB_eval/flickr_new/
