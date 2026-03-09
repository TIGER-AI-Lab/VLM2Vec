#!/bin/bash
# NOTE: replace ... with actual paths (keep system PATH/LD_LIBRARY_PATH)
echo "conda location: $(which conda)"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# export HF_DATASETS_CACHE=
# export HF_HOME=
# export WANDB_DISABLED=false
# export WANDB_PROJECT=
# export WANDB_API_KEY=
# export HUGGING_FACE_HUB_TOKEN=
# export WANDB_RUN_GROUP=
export EXP_NAME=Qwen2_5Omni_3B.audio.lora16.BS512.IB64.GCq8p8.NormTemp002.lr5e5.step5kwarm100

export WANDB_NAME=$EXP_NAME
export EXP_DIR=exps/output_model/$EXP_NAME
export WANDB_DIR=$EXP_DIR
echo $EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*

cd /data/mengrui/OLM2Vec/OLM2Vec
cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
torchrun \
--nproc_per_node=7 \
--master_port=2209 \
--max_restarts=0 \
train_omni.py \
--lora true \
--lora_r 16 \
--model_name /data/mengrui/.cache/huggingface/Qwen2.5-Omni-3B \
--model_type qwen2_5_omni \
--bf16 \
--pooling mean \
--normalize True \
--temperature 0.02 \
--loss_stage mixed \
--loss_alpha 0.5 \
--dataloader_num_workers 8 \
--dataset_config experiments/public/train/train_alltasks-v3.yaml \
--run_name \$EXP_NAME \
--output_dir \$EXP_DIR \
--grad_cache True \
--per_device_train_batch_size 8 \
--gc_q_chunk_size 4 \
--gc_p_chunk_size 4 \
--interleave_batch_size 8 \
--lr_scheduler_type linear \
--learning_rate 5e-5 \
--max_steps 15000 \
--warmup_steps 100 \
--save_steps 200 \
--logging_steps 1 \
--save_safetensors True \
--remove_unused_columns False \
--resume_from auto \
--report_to wandb \
2>&1 | tee \$EXP_DIR/train.log"

echo $cmd
eval $cmd
