#!/bin/bash
#conda activate /fsx/home/xyang/embed-env
export HF_DATASETS_CACHE=/fsx/home/ruimeng/data/.hfdata_cache
export HF_HOME=/fsx/home/ruimeng/data/.hfmodel_cache/
export HUGGING_FACE_HUB_TOKEN=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

cd /fsx/home/ruimeng/project/VLM2Vec

# Define the list of specific experiment directories you want to process
#SUBSET_NAMES="ImageNet-1K N24News CIFAR-100 HatefulMemes VOC2007 SUN397 ImageNet-A ImageNet-R ObjectNet Country211 VisDial CIRR FashionIQ VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA Wiki-SS-NQ OVEN EDIS OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA ScienceQA Visual7W VizWiz GQA TextVQA"
#SUBSET_NAMES="WebQA Wiki-SS-NQ OVEN EDIS"
#SUBSET_NAMES="ImageNet-1K N24News CIFAR-100 HatefulMemes VOC2007 SUN397 ImageNet-A ImageNet-R ObjectNet Country211"
#SUBSET_NAMES="OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA ScienceQA Visual7W VizWiz GQA TextVQA"
#SUBSET_NAMES="VisDial CIRR FashionIQ VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA Wiki-SS-NQ OVEN EDIS"
#SUBSET_NAMES="MSCOCO RefCOCO-Matching Visual7W-Pointing"

SUBSET_LIST=(
#  "ImageNet-1K N24News HatefulMemes VOC2007 SUN397 Place365 ImageNet-A ImageNet-R ObjectNet Country211"
#  "OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA ScienceQA Visual7W VizWiz GQA TextVQA"
#  "VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA FashionIQ Wiki-SS-NQ OVEN EDIS"
#  "MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing"

#"ImageNet-1K N24News"
#"HatefulMemes VOC2007 SUN397"
#"Place365 ImageNet-A"
#"ImageNet-R ObjectNet Country211"
#"OK-VQA A-OKVQA"
#"DocVQA InfographicsVQA ChartQA"
#"ScienceQA Visual7W"
#"VizWiz GQA TextVQA"

#"Wiki-SS-NQ"
#"VisualNews_t2i"
#"MSCOCO_t2i"
#"FashionIQ OVEN EDIS"
#"MSCOCO Visual7W-Pointing"
#"RefCOCO RefCOCO-Matching"
#"VisDial CIRR NIGHTS"
#"WebQA VisualNews_i2t MSCOCO_i2t"

  "ImageNet-1K N24News HatefulMemes VOC2007 SUN397"
  "Place365 ImageNet-A ImageNet-R ObjectNet Country211"
  "OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA"
  "ScienceQA Visual7W VizWiz GQA TextVQA"
  "VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t "
  "FashionIQ Wiki-SS-NQ"
  "NIGHTS WebQA OVEN EDIS"
  "MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing"
)

BATCH_SIZE=32
EXPERIMENTS_LIST=(
     # llava-next
#     "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_mistral-1.lora8.mmeb20_sub50k.bs256pergpu32.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#     "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_mistral-2.lora8.mmeb20_sub50k.bs1024pergpu128.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-1500/"
     "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_mistral-2.lora8.mmeb20_sub50k.bs1024pergpu128.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-1000/"

#     "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_vicuna-1.lora8.mmeb20_sub50k.bs256pergpu32.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#     "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_vicuna-1.lora8.mmeb20_sub50k.bs256pergpu32.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-1000/"
#     "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_mistral-2.lora8.mmeb20_sub50k.bs1024pergpu128.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-1000/"
#     "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_vicuna-2.lora8.mmeb20_sub50k.bs1024pergpu128.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-1000/"

#    #bs
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-bs1024.fullmodel.mmeb20_sub50k.bs1024pergpu128.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-bs2048.fullmodel.mmeb20_sub50k.bs2048pergpu256.GCq3p3.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    #task
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-taskVQA.fullmodel.mmeb20_sub50k.bs64pergpu8.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-taskRET.fullmodel.mmeb20_sub50k.bs64pergpu8.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-taskCLS.fullmodel.mmeb20_sub50k.bs64pergpu8.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    #lora
#    "/fsx/sfr/data/MMEB_exp/mmeb004-lora8.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    "/fsx/sfr/data/MMEB_exp/mmeb004-lora32.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-lora4.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-lora8_bs1k.mmeb20_sub50k.bs1024pergpu128.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-1000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-lora8_bs1k.mmeb20_sub50k.bs1024pergpu128.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    #maxlen
#    "/fsx/sfr/data/MMEB_exp/mmeb004-len128.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len128crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-len512.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len512crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#
#    #step
#    "/fsx/sfr/data/MMEB_exp/mmeb004-step1k.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step1kwarm50.8H100/checkpoint-1000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-step4k.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step4kwarm200.8H100/checkpoint-4000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-step8k.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step8kwarm400.8H100/checkpoint-8000/"
##    #crop
#    "/fsx/sfr/data/MMEB_exp/mmeb004-crop1.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop1.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    "/fsx/sfr/data/MMEB_exp/mmeb004-crop2.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop2.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-crop9.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq2p2.phi35.NormTemp002.len256crop9.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-crop16.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq1p1.phi35.NormTemp002.len256crop16.lr2e5.step2kwarm100.8H100/checkpoint-2000/"
    # data size
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-data25k.fullmodel.mmeb20_sub25k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step4kwarm200.8H100/checkpoint-4000/"
#    "/fsx/home/ruimeng/runs/mmeb/mmeb004-data100k.fullmodel.mmeb20_sub100k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step4kwarm200.8H100/checkpoint-4000/"
)

GPU_IDS=(0 1 2 3 4 5 6 7)
#GPU_IDS=(0)
NUM_GPUS=${#GPU_IDS[@]}  # Get the number of GPUs based on the array length

# Counter for tracking GPU assignment
gpu_index=0

print_success() {
    echo -e "\e[32mSUCCESS: $1\e[0m"
}
print_error() {
    echo -e "\e[31mERROR: $1\e[0m"
}
print_warning() {
    echo -e "\e[33mWARNING: $1\e[0m"
}
# Function to run an experiment on a specific GPU
run_experiment() {
    local checkpoint_path=$1
    local gpu_id=$2
    local subset_id=$3

    local ckpt_name=$(basename "$checkpoint_path")
    local exp_dir=$(dirname "$checkpoint_path")
    local exp_basename=$(basename "$exp_dir")
    echo $gpu_id $subset_id
#    echo $exp_dir
    echo $exp_basename
#    local output_path="/fsx/home/ruimeng/runs/mmeb/$exp_basename/$ckpt_name"
#    echo $output_dir
#    mkdir -p $output_path

    if [[ "$exp_basename" == *lora* ]]; then
        lora=" --lora"
        print_warning "Extracted lora: $lora"
    else
        lora=" "
        print_warning "lora not found in exp name"
    fi
    # Extract 'lenXXX' and 'cropXX' from the directory name using regex
    if [[ "$exp_basename" =~ len([0-9]+) ]]; then
        max_len="${BASH_REMATCH[1]}"
        echo "Extracted max_len: $max_len"
    else
        print_error "max_len not found in exp name, does it include the word 'len'?"
        return 1  # Exit the function with a non-zero status
    fi
    if [[ "$exp_basename" =~ crop([0-9]+) ]]; then
        num_crop="${BASH_REMATCH[1]}"
        echo "Extracted num_crop: $num_crop"
    else
        print_error "num_crop not found in the exp name, does it include the word 'crop'?"
        return 1  # Exit the function with a non-zero status
    fi
    # Build the command string
    cmd="CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=1 --master_port=229$gpu_id --max_restarts=0 eval.py --model_name llava-hf/llava-v1.6-mistral-7b-hf --processor_name llava-hf/llava-v1.6-mistral-7b-hf --model_backbone llava $lora --checkpoint_path $checkpoint_path --encode_output_path $checkpoint_path --max_len $max_len --num_crops $num_crop --pooling eos --normalize True --dataset_name EmbVision/MMEB_Test_1K_New --subset_name ${SUBSET_LIST[$subset_id]} --dataset_split test --per_device_eval_batch_size $BATCH_SIZE --image_dir /fsx/sfr/data/MMEB/MMEB_test/MMEB_Test_1K_New/images/"

    echo "Running on GPU $gpu_id: $cmd"
    eval $cmd  # Execute the command
}

# Loop through the list of specific experiment checkpoint directories
for checkpoint_path in "${EXPERIMENTS_LIST[@]}"; do
    echo "Processing $checkpoint_path"
    if [ -d "$checkpoint_path" ]; then
        # Get the current GPU ID from the GPU_IDS array

        for subset_id in "${!SUBSET_LIST[@]}"; do
            gpu_id=${GPU_IDS[$gpu_index]}
            # Run the experiment on the current GPU
#            echo $gpu_id $subset_id
            run_experiment "$checkpoint_path" $gpu_id $subset_id &

            # Increment GPU index and reset after reaching the maximum number of GPUs
            gpu_index=$(( (gpu_index + 1) % NUM_GPUS ))

            # If we completed a batch (all GPUs have been used), wait for the batch to finish
            if [ $gpu_index -eq 0 ]; then
                echo "Waiting for the current batch of jobs to finish..."
                wait  # Wait for all background jobs to finish before starting the next batch
            fi
        done
    else
        echo "Checkpoint directory $checkpoint_path does not exist"
    fi
done

# Wait for any remaining background jobs before exiting
wait
echo "All batches are complete."
