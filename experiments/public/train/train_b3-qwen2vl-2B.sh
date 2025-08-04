#!/bin/bash
# NOTE: replace ... with actual paths
export LD_LIBRARY_PATH=...
export PATH=...
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

export HF_DATASETS_CACHE=...
export HF_HOME=...
export WANDB_DISABLED=false
export WANDB_PROJECT=...
export WANDB_API_KEY=...
export HUGGING_FACE_HUB_TOKEN=...
export WANDB_PROJECT=...
export WANDB_RUN_GROUP=...
export EXP_NAME=Qwen2vl_2B.B3.lora16.TempTrain.lr1e4

export WANDB_NAME=$EXP_NAME
export EXP_DIR=.../$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cd PATH_TO_VLM2VEC_REPO
cmd="CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=2207 --max_restarts=0 train.py --lora --lora_r 16 --model_name Qwen/Qwen2-VL-2B-Instruct --bf16 --pooling eos --normalize True --temperature 0.02 --dataloader_num_workers 8 --dataset_config experiments/train/train_b3-2b.yaml --run_name $EXP_NAME --output_dir $EXP_DIR --grad_cache True --per_device_train_batch_size 128 --gc_q_chunk_size 8 --gc_p_chunk_size 8 --gc_dynamic_limit 64 --interleave_batch_size 1 --lr_scheduler_type linear --learning_rate 1e-4 --max_steps 100 --warmup_steps 10 --save_steps 10 --logging_steps 1 --save_safetensors True --remove_unused_columns False --resume_from auto --resize_use_processor True --chuck_size 32 --interleave_datasets --report_to wandb 2>&1 | tee $EXP_DIR/train.log"

echo $cmd
eval $cmd
