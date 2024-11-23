export HF_DATASETS_CACHE=/fsx/home/ruimeng/data/.hfdata_cache
export HF_HOME=/fsx/home/ruimeng/data/.hfmodel_cache/
export WANDB_DISABLED=false
export WANDB_PROJECT=unified_embedding
export WANDB_API_KEY=local-d64a4127e8d4a1782aedbb72e76080b3dfbf89dd
export WANDB_BASE_URL=https://salesforceairesearch.wandb.io
export HUGGING_FACE_HUB_TOKEN=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
export WANDB_PROJECT=mmeb
export WANDB_RUN_GROUP=train
export EXP_NAME=mmeb005-llava16_mistral-1.lora8.mmeb20_sub50k.bs256pergpu32.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100


export WANDB_NAME=$EXP_NAME
export EXP_DIR=/fsx/home/ruimeng/runs/mmeb/$EXP_NAME
export WANDB_DIR=$EXP_DIR


mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*
cd /fsx/home/ruimeng/project/VLM2Vec
# mmeb-train 20
CLS_DATASETS="ImageNet_1K N24News HatefulMemes VOC2007 SUN397"
VQA_DATASETS="OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W"
RET_DATASETS="VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA"
VG_DATASETS="MSCOCO"
TRAIN_DATASETS="$CLS_DATASETS $VQA_DATASETS $RET_DATASETS $VG_DATASETS"
export IMAGE_DIR=/fsx/sfr/data/MMEB

cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2207 --max_restarts=0 train.py --lora --lora_r 8 --model_name llava-hf/llava-v1.6-mistral-7b-hf --model_backbone llava --bf16 --pooling eos --normalize True --temperature 0.02 --gradient_accumulation_steps 1 --dataset_name EmbVision/MMEB_Train --subset_name $TRAIN_DATASETS --num_sample_per_subset 50000 --image_dir $IMAGE_DIR --run_name $EXP_NAME --output_dir $EXP_DIR --max_len 256 --num_crops 4 --grad_cache True --per_device_train_batch_size 32 --gc_q_chunk_size 2 --gc_p_chunk_size 2 --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 2000 --warmup_steps 100 --save_steps 100 --logging_steps 1 --save_safetensors False 2>&1 | tee $EXP_DIR/train.log"
echo $cmd
eval $cmd


