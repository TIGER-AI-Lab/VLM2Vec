from huggingface_hub import login
from datasets import Dataset
import json
import csv


HF_API_TOKEN = ""
HF_Repo = "ziyjiang/MMEB-Pro-Train"
Subset_Name = "NExT-QA"

MapFilePath = "/home/ziyan/MMEB_Pro/NExT-QA/map_vid_vidorID.json"
TrainFilePath = "/home/ziyan/MMEB_Pro/NExT-QA/train.csv"

TrainDataset = []
Mapping = {}

with open(MapFilePath) as f:
    Mapping = json.load(f)

with open(TrainFilePath, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        question = row["question"]
        answer = row["answer"]
        video_path = Mapping[row["video"]]
        TrainDataset.append({
            "instruction": "Answer the question below based on the provided video.",
            "qry_text": question,
            "qry_video_path": Subset_Name + "/" + video_path,
            "pos_text": answer,
            "pos_video_path": "",
        })

train_data_hf = Dataset.from_list(TrainDataset)
train_data_hf.push_to_hub(HF_Repo, Subset_Name, split="train")
print("Uploaded to huggingface repo!")
