from transformers import HfArgumentParser

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.dataset import FlickrDataset
from src.collator import EvalCollator
from src.model_utils import load_processor
from src.utils import batch_to_device

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

    processor = load_processor(model_args)

    eval_img_dataset = FlickrDataset(
        modality="image",
        model_backbone=model_args.model_backbone,
        image_resolution="high",
    )
    eval_txt_dataset = FlickrDataset(
        modality="text",
        model_backbone=model_args.model_backbone,
        image_resolution="high",
    )
    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )

    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    encode_img_path = os.path.join(data_args.encode_output_path, f"flickr_image_1K")
    encode_txt_path = os.path.join(data_args.encode_output_path, f"flickr_text_1K")

    if not (os.path.exists(encode_img_path) and os.path.exists(encode_txt_path)):
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

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_img_loader, desc="Encode image"):
                batch = batch_to_device(batch, training_args.device)
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output = model(qry=batch)
                encoded_tensor.append(output["qry_reps"].cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_img_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_img_dataset.image_names), f)

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_txt_loader, desc="Encode text"):
                batch = batch_to_device(batch, training_args.device)
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
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
    similarity_matrix = torch.matmul(img_tensor, txt_tensor.T)
    recall_at_k = {1: 0, 5: 0, 10: 0}
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    for idx, file_name in enumerate(i2t_name):
        top_k_indices = sorted_indices[idx, :10]  # Get top-10 indices
        top_k_file_names = [t2i_name[i.item()] for i in top_k_indices]
        for k in [1, 5, 10]:
            if file_name in top_k_file_names[:k]:
                recall_at_k[k] += 1

    for k in [1, 5, 10]:
        recall_at_k[k] = recall_at_k[k] / len(i2t_name)
        print(f"\033[91m Recall@{k}: {recall_at_k[k]:.4f}\033[0m")


    # T -> I
    similarity_matrix = torch.matmul(txt_tensor, img_tensor.T)
    recall_at_k = {1: 0, 5: 0, 10: 0}
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    for idx, file_name in enumerate(t2i_name):
        top_k_indices = sorted_indices[idx, :10]
        top_k_file_names = [i2t_name[i.item()] for i in top_k_indices]
        for k in [1, 5, 10]:
            if file_name in top_k_file_names[:k]:
                recall_at_k[k] += 1

    for k in [1, 5, 10]:
        recall_at_k[k] = recall_at_k[k] / len(t2i_name)
        print(f"\033[91m Recall@{k}: {recall_at_k[k]:.4f}\033[0m")


if __name__ == "__main__":
    main()
