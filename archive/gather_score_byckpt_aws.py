import os
import json
import re

# Define the datasets
datasets = [
    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
    "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA",
    "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS",
    "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"
]

# Define the root directory containing the experiment directories
checkpoint_paths = [
    # HF
    # "/fsx/home/ruimeng/runs/test/hf-VLM2Vec-LLaVa-Next-lateprocess",

    # qwen
    "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-3-lateprocess-mid_res-flashattn-leftpad.lora8.mmeb20_sub100k.bs256pergpu32.GCq8p8_OOM.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1000",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-2-lateprocess-mid_res.lora8.mmeb20_sub100k.bs256pergpu32.GCq8p8_OOM.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1000",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-1-lateprocess-low_res.lora8.mmeb20_sub100k.bs1024pergpu128.GCq32p32.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb006-qwen2vl-2B-1-lateprocess-low_res.lora8.mmeb20_sub100k.bs1024pergpu128.GCq32p32.NormTemp002.lr2e5.step2kwarm100.8H100/checkpoint-1500/",

    # late-process
    # "/fsx/home/ruimeng/runs/mmeb/mmeb006-llava16_mistral-4-lateprocess-highres.lora8.mmeb20_sub100k.bs1024pergpu128.GCq1p1.NormTemp002.GradClip05.lr2e5.step2kwarm100.8H100/checkpoint-500/",
    # this didn't converge
    # "/fsx/home/ruimeng/runs/mmeb/mmeb006-llava16_mistral-3-lateprocess-highres.lora8.mmeb20_sub100k.bs1024pergpu128.len2048.GCq1p1.NormTemp002.lr2e5.step1kwarm50.8H100/checkpoint-50/",

    # "/fsx/home/ruimeng/runs/mmeb/mmeb006-phi35v-1-lateprocess-highres.lora8.mmeb20_sub100k.bs1024pergpu128.GCq1p1.maxlen2k.crop4.NormTemp002.lr2e5.step1kwarm50.8H100/checkpoint-500/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb006-phi35v-1-lateprocess-highres.lora8.mmeb20_sub100k.bs1024pergpu128.GCq1p1.maxlen2k.crop4.NormTemp002.lr2e5.step1kwarm50.8H100/checkpoint-500/eval_v1/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb006-phi35v-1-lateprocess-highres.lora8.mmeb20_sub100k.bs1024pergpu128.GCq1p1.maxlen2k.crop4.NormTemp002.lr2e5.step1kwarm50.8H100/checkpoint-300/",

    # llava-next
    # "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_mistral-3.lora8.mmeb20_sub100k-1344.bs1024pergpu128.GCq1p1.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-1000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_mistral-3.lora8.mmeb20_sub100k-1344.bs1024pergpu128.GCq1p1.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-1400/",
     # "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_mistral-1.lora8.mmeb20_sub50k.bs256pergpu32.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
     # "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_mistral-2.lora8.mmeb20_sub50k.bs1024pergpu128.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
     # "/fsx/home/ruimeng/runs/mmeb/mmeb005-e5v-1.lora8.mmeb20_sub50k.bs1024pergpu128.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",

    # "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_vicuna-1.lora8.mmeb20_sub50k.bs256pergpu32.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb005-llava16_vicuna-2.lora8.mmeb20_sub50k.bs1024pergpu128.GCq2p2.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",

    # scale-up
    # "/fsx/home/ruimeng/runs/mmeb/mmeb005-scale002.lora8.mmeb17_sub100k_NoMSCOCO.bs1024pergpu128.GCq2p2.phi35.NormTemp002.len256crop9.lr5e5.step5kwarm200.8H100/checkpoint-1500/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb005-scale001.lora8.mmeb20_sub100k.bs1024pergpu128.GCq2p2.phi35.NormTemp002.len256crop9.lr2e5.step5kwarm200.8H100/checkpoint-1500/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb005-scale001.lora8.mmeb20_sub100k.bs1024pergpu128.GCq2p2.phi35.NormTemp002.len256crop9.lr2e5.step5kwarm200.8H100/checkpoint-2500/",

    # "/fsx/home/ruimeng/runs/mmeb/mmeb005-scale002-1.lora8.mmeb17_sub100k_NoMSCOCO.bs1024pergpu128.GCq2p2.phi35.NormTemp002.len256crop9.lr2e5.step2kwarm100.8H100/checkpoint-1000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb005-scale002-1.lora8.mmeb17_sub100k_NoMSCOCO.bs1024pergpu128.GCq2p2.phi35.NormTemp002.len256crop9.lr2e5.step2kwarm100.8H100/checkpoint-1500/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb005-scale002-1.lora8.mmeb17_sub100k_NoMSCOCO.bs1024pergpu128.GCq2p2.phi35.NormTemp002.len256crop9.lr2e5.step2kwarm100.8H100/checkpoint-2000/",

    # batch size
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-bs1024.fullmodel.mmeb20_sub50k.bs1024pergpu128.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # # task
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-taskVQA.fullmodel.mmeb20_sub50k.bs64pergpu8.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-taskRET.fullmodel.mmeb20_sub50k.bs64pergpu8.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-taskCLS.fullmodel.mmeb20_sub50k.bs64pergpu8.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # # lora
    # "/fsx/sfr/data/MMEB_exp/mmeb004-lora8.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/sfr/data/MMEB_exp/mmeb004-lora32.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/sfr/data/MMEB_exp/mmeb004-lora8_bs1k.mmeb20_sub50k.bs1024pergpu128.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # # maxlen
    # "/fsx/sfr/data/MMEB_exp/mmeb004-len128.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len128crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-len512.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len512crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # # step
    # "/fsx/sfr/data/MMEB_exp/mmeb004-step1k.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step1kwarm50.8H100/checkpoint-1000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-step4k.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step4kwarm200.8H100/checkpoint-4000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-step8k.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step8kwarm400.8H100/checkpoint-8000/",
    # # crop
    # "/fsx/sfr/data/MMEB_exp/mmeb004-crop1.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop1.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/sfr/data/MMEB_exp/mmeb004-crop2.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop2.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-crop9.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq2p2.phi35.NormTemp002.len256crop9.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-crop16.fullmodel.mmeb20_sub50k.bs256pergpu32.GCq1p1.phi35.NormTemp002.len256crop16.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # data size
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-lora8_bs1k.mmeb20_sub50k.bs1024pergpu128.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-1000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-lora4.mmeb20_sub50k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step2kwarm100.8H100/checkpoint-2000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-data25k.fullmodel.mmeb20_sub25k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step4kwarm200.8H100/checkpoint-4000/",
    # "/fsx/home/ruimeng/runs/mmeb/mmeb004-data100k.fullmodel.mmeb20_sub100k.bs256pergpu32.GCq4p4.phi35.NormTemp002.len256crop4.lr2e5.step4kwarm200.8H100/checkpoint-4000/",
]


# Function to extract step number from checkpoint directory name
def extract_step(checkpoint_name):
    match = re.search(r'checkpoint-(\d+)', checkpoint_name)
    return int(match.group(1)) if match else float('inf')


# Dictionary to hold all gathered scores, organized by experiment
gathered_scores_by_exp = {}

# Loop through checkpoint directories
for checkpoint_path in checkpoint_paths:
    step = extract_step(checkpoint_path)
    experiment_dir = checkpoint_path.split("/")[-3]

    # Check if it is a checkpoint directory, and a valid checkpoint dir
    if str.isdigit(str(step)):
        # Initialize a dictionary to store scores for this checkpoint
        checkpoint_scores = {"experiment": experiment_dir, "checkpoint": str(step)}
    else:
        checkpoint_scores = {"experiment": experiment_dir, "checkpoint": "default"}

    # Go through each dataset and check if the corresponding score file exists
    for dataset in datasets:
        score_file = os.path.join(checkpoint_path, f"{dataset}_score.json")  # Score file named like DatasetName_score.json

        # Check if the score file exists
        if os.path.isfile(score_file):
            with open(score_file, "r") as f:
                score_data = json.load(f)  # Load the score JSON
                checkpoint_scores[dataset] = score_data.get("acc", "N/A")  # Assuming 'acc' is the key for accuracy
        else:
            checkpoint_scores[dataset] = "N/A"  # If no score file, set to 'N/A'

    # Append the scores for this checkpoint to the respective experiment group
    gathered_scores_by_exp[experiment_dir] = checkpoint_scores



print('\n' * 5)
# Print gathered scores in a comma-separated format
header = ["experiment", "checkpoint"] + datasets
print(",".join(header))  # Print header

for experiment, scores in gathered_scores_by_exp.items():
    row = [scores["experiment"], scores["checkpoint"]] + [str(scores[dataset]) for dataset in datasets]
    print(",".join(row))  # Print each row of scores



header = ["dataset"] + list(gathered_scores_by_exp.keys())
print(",".join(header))  # Print header
# Additional Block: Print results per experiment, transposed (dataset per row, step per column)
# Print dataset names in the first column, and the scores for each checkpoint in subsequent columns
for dataset in datasets:
    row = []
    for experiment, scores in gathered_scores_by_exp.items():
        row.append(str(scores[dataset]))
    print(",".join([dataset] + row))  # Print header




# header = ["dataset"] + list(gathered_scores_by_exp.keys())
# print(",".join(header))  # Print header
# # Additional Block: Print results per experiment, transposed (dataset per row, step per column)
# # Print dataset names in the first column, and the scores for each checkpoint in subsequent columns
# for dataset in datasets:
#     print(",".join([dataset, str(scores[dataset])]))
#     for experiment, scores in gathered_scores_by_exp.items():
#         print(f"\nResults for {experiment}:")
#
