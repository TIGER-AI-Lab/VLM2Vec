export PYTHONPATH=../VLM2Vec/:$PYTHONPATH
export OUTPUT_DIR=/home/ziyan/exp/test
#CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=22447 --max_restarts=0 train.py \
# --model_name microsoft/Phi-3.5-vision-instruct --bf16 --pooling last \
# --model_backbone phi \
# --dataset_name TIGER-Lab/MMEB-train \
# --subset_name ImageNet_1K N24News MSCOCO WebQA \
# --num_sample_per_subset 50000 \
# --image_dir /home/ziyan/ \
# --max_len 256 --num_crops 1 --output_dir $OUTPUT_DIR --logging_steps 1 \
# --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 20 \
# --warmup_steps 200 --save_steps 1000 --normalize True \
# --temperature 0.02 --per_device_train_batch_size 8 \
# --lora --lora_r 16 \
# --grad_cache True --gc_q_chunk_size 2 --gc_p_chunk_size 2

CUDA_VISIBLE_DEVICES=3 python train.py \
 --model_name microsoft/Phi-3.5-vision-instruct --bf16 --pooling last \
 --model_backbone phi \
 --dataset_name TIGER-Lab/MMEB-train \
 --subset_name ImageNet_1K N24News MSCOCO WebQA \
 --num_sample_per_subset 50000 \
 --image_dir /home/ziyan/ \
 --max_len 256 --num_crops 16 --output_dir $OUTPUT_DIR --logging_steps 1 \
 --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 20 \
 --warmup_steps 200 --save_steps 1000 --normalize True \
 --temperature 0.02 --per_device_train_batch_size 2 \
  --lora --lora_r 16
