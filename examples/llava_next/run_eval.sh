export PYTHONPATH=../VLM2Vec/:$PYTHONPATH

CUDA_VISIBLE_DEVICES=7 python eval.py \
  --model_name TIGER-Lab/VLM2Vec-LLaVa-Next \
  --image_dir /data/ziyan/MMEB \
  --encode_output_path /home/ziyan/MMEB_eval/ \
  --pooling eos --normalize True \
  --dataset_name TIGER-Lab/MMEB-eval \
  --dataset_split test \
  --subset_name ImageNet-1K  \
  --image_resolution high \
  --per_device_eval_batch_size 8

# ImageNet-1K accuracy: 0.748
