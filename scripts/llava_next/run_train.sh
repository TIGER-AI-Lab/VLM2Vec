export PYTHONPATH=/home/ziyan/VLM2Vec:$PYTHONPATH

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=22447 --max_restarts=0 train.py \
  --model_name llava-hf/llava-v1.6-mistral-7b-hf \
  --model_backbone llava_next \
  --output_dir /home/ziyan/exp/vlm2vec_llava_next_test \
  --bf16 --pooling last \
  --lora \
  --dataset_name TIGER-Lab/MMEB-train \
  --subset_name ImageNet_1K N24News HatefulMemes InfographicsVQA ChartQA Visual7W VisDial CIRR NIGHTS WebQA MSCOCO \
  --num_sample_per_subset 50000 \
  --image_dir /home/ziyan/ \
  --max_len 64 --num_crops 1 --logging_steps 1 \
  --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 2000 \
  --warmup_steps 200 --save_steps 1000 --normalize True \
  --temperature 0.02 --per_device_train_batch_size 1 \
  --grad_cache True --gc_q_chunk_size 2 --gc_p_chunk_size 2
