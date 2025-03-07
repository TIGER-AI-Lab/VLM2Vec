#!/bin/bash
export PYTHONPATH=/home/ziyan/VLM2Vec:$PYTHONPATH  # Change to your own path

cd project/VLM2Vec/
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=22447 --max_restarts=0 train.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --output_dir runs/test/mmeb-qwen-test \
  --bf16 --pooling last \
  --lora \
  --dataset_name TIGER-Lab/MMEB-train \
  --split_name original \
  --subset_name ImageNet_1K N24News VisDial \
  --num_sample_per_subset 10000 \
  --image_dir data/MMEB/MMEB-train \
  --image_resolution high --max_len 4096 \
  --logging_steps 1 \
  --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 2000 \
  --warmup_steps 200 --save_steps 1000 --normalize True \
  --temperature 0.02 --per_device_train_batch_size 32 \
  --grad_cache True --gc_q_chunk_size 4 --gc_p_chunk_size 4 \
  --save_safetensors False --remove_unused_columns False \
  --report_to none
