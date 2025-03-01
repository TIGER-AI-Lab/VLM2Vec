from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login
from datasets import Dataset

HF_API_TOKEN = ""
HF_Repo = "ziyjiang/MMEB-Pro-Train"
Subset_Name = "MSR-VTT"

TrainDataset = []

dataset = load_dataset('AlexZigma/msr-vtt')
for row in tqdm(dataset['train']):
    video_name = row['video_id']
    caption = row['caption']
    TrainDataset.append({
        "instruction": "Retrieve the caption for the provided video clip.",
        "qry_text": "",
        "qry_video_path": Subset_Name + "/" + video_name + ".mp4",
        "pos_text": caption,
        "pos_video_path": "",
    })

train_data_hf = Dataset.from_list(TrainDataset)
train_data_hf.push_to_hub(HF_Repo, Subset_Name, split="train")
print("Uploaded to huggingface repo!")
