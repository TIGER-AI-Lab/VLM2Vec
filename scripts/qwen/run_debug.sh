export PYTHONPATH=/home/ziyan/VLM2Vec:$PYTHONPATH  # Change to your own path

CUDA_VISIBLE_DEVICES=7 python train.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --model_backbone qwen \
  --output_dir /home/ziyan/exp/vlm2vec_llava_next_test \
  --bf16 --pooling last \
  --lora \
  --dataset_name TIGER-Lab/MMEB-train \
  --subset_name ImageNet_1K N24News VisDial \
  --num_sample_per_subset 100 \
  --image_dir /home/ziyan/ \
  --max_len 32 --logging_steps 1 \
  --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 2000 \
  --warmup_steps 200 --save_steps 1000 --normalize True \
  --temperature 0.02 --per_device_train_batch_size 4
