#!/bin/bash
# This script is intended only for a quick demo, using a small amount of training data and a small batch size.
export PYTHONPATH=/home/ziyan/VLM2Vec:$PYTHONPATH  # Change to your own path
export HF_HOME=~/.cache/huggingface
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=22447 --max_restarts=0 train.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --output_dir ./vlm2vec-qwen-test \
  --bf16 --pooling last \
  --lora \
  --dataset_name TIGER-Lab/MMEB-train \
  --split_name original \
  --subset_name ImageNet_1K N24News VisDial \
  --num_sample_per_subset 10000 \
  --image_dir /home/ziyan/ \
  --image_resolution high --max_len 4096 \
  --logging_steps 1 \
  --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 200 \
  --warmup_steps 200 --save_steps 10 --normalize True \
  --temperature 0.02 --per_device_train_batch_size 8 \
  --grad_cache True --gc_q_chunk_size 2 --gc_p_chunk_size 2 \
  --save_safetensors False --remove_unused_columns False \
  --report_to none


# resume training
#CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=22447 --max_restarts=0 train.py \
#  --model_name Qwen/Qwen2-VL-2B-Instruct \
#  --output_dir ./vlm2vec-qwen-test \
#  --resume_from_checkpoint ./vlm2vec-qwen-test/checkpoint-20 \
#  --bf16 --pooling last \
#  --lora \
#  --dataset_name TIGER-Lab/MMEB-train \
#  --split_name original \
#  --subset_name ImageNet_1K N24News VisDial \
#  --num_sample_per_subset 10000 \
#  --image_dir /home/ziyan/ \
#  --image_resolution high --max_len 4096 \
#  --logging_steps 1 \
#  --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 200 \
#  --warmup_steps 200 --save_steps 10 --normalize True \
#  --temperature 0.02 --per_device_train_batch_size 8 \
#  --grad_cache True --gc_q_chunk_size 2 --gc_p_chunk_size 2 \
#  --save_safetensors False --remove_unused_columns False \
#  --report_to none
