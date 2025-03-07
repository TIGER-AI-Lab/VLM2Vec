import os
from huggingface_hub import login
from datasets import Dataset


HF_API_TOKEN = ""
HF_Repo = "ziyjiang/MMEB-Pro-Train"
Subset_Name = "HMDB51"

TrainFileDir = "/home/ziyan/MMEB_Pro/HMDB51/testTrainMulti_7030_splits"
VideoDir = "/home/ziyan/MMEB_Pro/HMDB51"

TrainDataset = []


for filename in os.listdir(TrainFileDir):
    if filename.endswith('1.txt'):
        file_path = os.path.join(TrainFileDir, filename)
        action = filename.split('_test_split')[0]
        with open(file_path, 'r') as file:
            tt = 0
            for line in file:
                parts = line.strip().split(maxsplit=1)
                video_name = parts[0]
                split = int(parts[1])
                if split == 1:
                    tt += 1
                    video_path = os.path.join(Subset_Name, action, video_name)
                    TrainDataset.append({
                        "instruction": "Identify the action shown in the video.",
                        "qry_text": "",
                        "qry_video_path": video_path,
                        "pos_text": action,
                        "pos_video_path": "",
                    })
                    assert os.path.exists(os.path.join(VideoDir, action, video_name))

train_data_hf = Dataset.from_list(TrainDataset)
train_data_hf.push_to_hub(HF_Repo, Subset_Name, split="train")
print("Uploaded to huggingface repo!")
