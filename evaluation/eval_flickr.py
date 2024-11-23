from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor
from src.model import MMEBModel
from src.dataset import FlickrDataset
from src.collator import EvalCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from eval_utils import get_pred


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    processor = AutoProcessor.from_pretrained(
        model_args.processor_name if model_args.processor_name else model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )
    processor.tokenizer.padding_side = "right"
    eval_img_dataset = FlickrDataset(
        modality="image",
    )
    eval_txt_dataset = FlickrDataset(
        modality="text",
    )

    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )
    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    eval_img_loader = DataLoader(
        eval_img_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=eval_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    eval_txt_loader = DataLoader(
        eval_txt_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=eval_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    encode_img_path = os.path.join(data_args.encode_output_path, f"flickr_image_1K-crop{model_args.num_crops}")
    encode_txt_path = os.path.join(data_args.encode_output_path, f"flickr_text_1K-crop{model_args.num_crops}")

    encoded_tensor = []
    with torch.no_grad():
        for batch in tqdm(eval_img_loader, desc="Encode image"):
            batch = {key: value.to(training_args.device) for key, value in batch.items()}
            output = model(qry=batch)
            encoded_tensor.append(output["qry_reps"].cpu().detach().float().numpy())
    encoded_tensor = np.concatenate(encoded_tensor)
    with open(encode_img_path, 'wb') as f:
        pickle.dump((encoded_tensor, eval_img_dataset.image_names), f)

    encoded_tensor = []
    with torch.no_grad():
        for batch in tqdm(eval_txt_loader, desc="Encode text"):
            batch = {key: value.to(training_args.device) for key, value in batch.items()}
            output = model(qry=batch)
            encoded_tensor.append(output["qry_reps"].cpu().detach().float().numpy())
    encoded_tensor = np.concatenate(encoded_tensor)
    with open(encode_txt_path, 'wb') as f:
        pickle.dump((encoded_tensor, eval_txt_dataset.image_names), f)

    with open(encode_img_path, 'rb') as f:
        img_tensor, i2t_name = pickle.load(f)
        img_tensor = torch.from_numpy(img_tensor)
    with open(encode_txt_path, 'rb') as f:
        txt_tensor, t2i_name = pickle.load(f)
        txt_tensor = torch.from_numpy(txt_tensor)

    # I -> T
    acc = 0
    similarity_matrix = torch.matmul(img_tensor, txt_tensor.T)
    most_similar_indices = torch.argmax(similarity_matrix, dim=1)
    for idx, file_name in enumerate(i2t_name):
        pred_file_name = t2i_name[most_similar_indices[idx].item()]
        if file_name == pred_file_name:
            acc += 1
    print(f"\033[91m accuracy: {acc/len(i2t_name)}\033[0m")

    # T -> I
    acc = 0
    similarity_matrix = torch.matmul(txt_tensor, img_tensor.T)
    most_similar_indices = torch.argmax(similarity_matrix, dim=1)
    for idx, file_name in enumerate(t2i_name):
        pred_file_name = i2t_name[most_similar_indices[idx].item()]
        if file_name == pred_file_name:
            acc += 1
    print(f"\033[91m accuracy: {acc/len(t2i_name)}\033[0m")


if __name__ == "__main__":
    main()
