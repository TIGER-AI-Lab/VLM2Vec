export HF_DATASETS_CACHE=/fsx/home/ruimeng/data/.hfdata_cache
export HF_HOME=/fsx/home/ruimeng/data/.hfmodel_cache/
export WANDB_DISABLED=false
export WANDB_PROJECT=unified_embedding
export WANDB_API_KEY=local-d64a4127e8d4a1782aedbb72e76080b3dfbf89dd
export WANDB_BASE_URL=https://salesforceairesearch.wandb.io
export HUGGING_FACE_HUB_TOKEN=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
export WANDB_PROJECT=mmeb
export WANDB_RUN_GROUP=train
export EXP_NAME=mmeb005-scale001-continual2500.lora8.mmeb20_sub100k.bs1024pergpu128.GCq2p2.phi35.NormTemp002.len256crop9.lr2e5.step3kwarm100.8H100

export WANDB_NAME=$EXP_NAME
export EXP_DIR=/fsx/home/ruimeng/runs/mmeb/$EXP_NAME
export WANDB_DIR=$EXP_DIR

mkdir -p $EXP_DIR/wandb
rm -rf $EXP_DIR/wandb/*
# mmeb-train 20
CLS_DATASETS="ImageNet_1K N24News HatefulMemes VOC2007 SUN397"
VQA_DATASETS="OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W"
RET_DATASETS="VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA"
VG_DATASETS="MSCOCO"
TRAIN_DATASETS="$CLS_DATASETS $VQA_DATASETS $RET_DATASETS $VG_DATASETS"
export IMAGE_DIR=/fsx/sfr/data/MMEB

cd /fsx/home/ruimeng/project/VLM2Vec/
cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2207 --max_restarts=0 train.py --lora --lora_r 8 --model_name microsoft/Phi-3.5-vision-instruct --processor_name microsoft/Phi-3.5-vision-instruct --model_backbone phi35v --bf16 --pooling eos --normalize True --temperature 0.02 --gradient_accumulation_steps 1 --dataset_name EmbVision/MMEB_Train --subset_name $TRAIN_DATASETS --num_sample_per_subset 100000 --image_dir $IMAGE_DIR --run_name $EXP_NAME --output_dir $EXP_DIR --max_len 256 --num_crops 9 --grad_cache True --per_device_train_batch_size 128 --gc_q_chunk_size 2 --gc_p_chunk_size 2 --lr_scheduler_type linear --learning_rate 1e-5 --max_steps 200 --warmup_steps 10 --save_steps 10 --logging_steps 1 --save_safetensors False 2>&1 | tee $EXP_DIR/train.log"
echo $cmd
eval $cmd
