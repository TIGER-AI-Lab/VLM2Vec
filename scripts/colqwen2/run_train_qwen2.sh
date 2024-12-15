OUTPUT_DIR="output/colqwen2"
torchrun --nproc_per_node=2 --master_port=22447 --max_restarts=0 train.py \
 --model_name Qwen/Qwen2-VL-7B-Instruct --bf16 --pooling last \
 --model_backbone colqwen2 \
 --dataset_name TIGER-Lab/MMEB-train \
 --subset_name ImageNet_1K N24News HatefulMemes InfographicsVQA ChartQA Visual7W VisDial CIRR NIGHTS WebQA MSCOCO \
 --num_sample_per_subset 10000 \
 --image_dir MMEB-train \
 --max_len 128 --num_crops 2 --output_dir $OUTPUT_DIR --logging_steps 1 \
 --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 2000 \
 --warmup_steps 200 --save_steps 1000 --normalize False \
 --temperature 0.02 --per_device_train_batch_size 2 \
 --lora --lora_r 8 \
 --grad_cache False --gc_q_chunk_size 2 --gc_p_chunk_size 2
