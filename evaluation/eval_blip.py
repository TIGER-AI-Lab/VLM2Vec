# https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_feature_extraction.ipynb
# https://medium.com/@enrico.randellini/image-and-text-features-extraction-with-blip-and-blip-2-how-to-build-a-multimodal-search-engine-a4ceabf51fbe
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor
from src.dataset import EvalDataset
from evaluation.collator import EvalCollator, BLIP2Collator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.eval_utils import get_pred, save_results, print_results
from lavis.models import load_model_and_preprocess

t2i_tasks = [
    "EDIS", "MSCOCO_t2i","VisDial","VisualNews_t2i","WebQA", "Wiki-SS-NQ", # retrieval
    ]
i2t_tasks = [
    "MSCOCO_i2t","VisualNews_i2t", # retrieval
    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211" # classification
    ]


def get_pred_blip(qry_t, tgt_t, mode="multimodal2text"):

    if mode == "multimodal2text":
        # Compute the dot product between each token in qry_t (shape 32, dim) and tgt_t (shape candidate_num, dim)
        # This results in a (32, candidate_num) array of scores
        scores = np.dot(qry_t, tgt_t.T)  # (32, dim) dot (candidate_num, dim).T -> (32, candidate_num)

        # Find the maximum score for each candidate across the 32 tokens
        max_scores = np.max(scores, axis=0)  # Max along the 32 tokens for each candidate (shape candidate_num)

        # The prediction is the index of the target with the highest maximum score
        pred = np.argmax(max_scores)

    elif mode == "text2multimodal":
        # Compute the dot product between qry_t (shape dim) and each of the 32 tokens in the target (candidate_num, 32, dim)
        # This results in a (candidate_num, 32) array of scores
        scores = np.dot(tgt_t, qry_t)  # (candidate_num, 32, dim) dot (dim) -> (candidate_num, 32)

        # Find the maximum score for each candidate across the 32 tokens
        max_scores = np.max(scores, axis=1)  # Max along the 32 tokens for each candidate (shape candidate_num)

        # The prediction is the index of the target with the highest maximum score
        pred = np.argmax(max_scores)

    return max_scores, pred


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    model, vis_processors, txt_processors = load_model_and_preprocess(name=model_args.model_name, model_type=model_args.model_type, is_eval=True, device=training_args.device)
    embedding_type = data_args.embedding_type
    eval_collator = BLIP2Collator(
        data_args=data_args,
        vis_processors=vis_processors,
        txt_processors=txt_processors
    )

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(data_args.subset_name):
        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
            continue

        eval_qry_dataset = EvalDataset(
            data_args=data_args,
            subset=subset,
            text_field="qry_text",
            img_path_field="qry_img_path",
        )
        eval_tgt_dataset = EvalDataset(
            data_args=data_args,
            subset=subset,
            text_field="tgt_text",
            img_path_field="tgt_img_path",
        )

        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_qry_loader, desc="Encode query"):
                samples, modes = batch
                for sample, mode in zip(samples, modes):
                    image_features, text_features = None, None
                    if sample["image"] is not None:
                        sample["image"] = sample["image"].to(training_args.device)
                        image_features = model.extract_features(sample, mode="image").image_embeds[0,0,:] # (dim,)
                    if sample["text_input"]:
                        text_features = model.extract_features(sample, mode="text").text_embeds[0,0,:] # (dim,)
                    if embedding_type=="unimodal":
                        if subset in t2i_tasks:
                            features = text_features
                        if subset in i2t_tasks:
                            features = image_features
                    elif embedding_type=="multimodal":
                        if image_features is None:
                            features = text_features
                        elif text_features is None:
                            features = image_features
                        else:
                            features = image_features + text_features
                    encoded_tensor.append(features.cpu().detach().float().numpy())
        with open(encode_qry_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                samples, modes = batch
                for sample, mode in zip(samples, modes):
                    image_features, text_features = None, None
                    if sample["image"] is not None:
                        sample["image"] = sample["image"].to(training_args.device)
                        image_features = model.extract_features(sample, mode="image").image_embeds[0,0,:] # (dim,)
                    if sample["text_input"]:
                        text_features = model.extract_features(sample, mode="text").text_embeds[0,0,:] # (dim,)
                    if embedding_type=="unimodal":
                        if subset in t2i_tasks:
                            features = image_features
                        if subset in i2t_tasks:
                            features = text_features
                    elif embedding_type=="multimodal":
                        if image_features is None:
                            features = text_features
                        elif text_features is None:
                            features = image_features
                        else:
                            features = image_features + text_features
                    encoded_tensor.append(features.cpu().detach().float().numpy())
        with open(encode_tgt_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

    results = {}
    for subset in tqdm(data_args.subset_name, desc="calculate score"):
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)
        qry_dict, tgt_dict = {}, {}
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_dict[(text, img_path)] = qry_t
        for tgt_t, tt in zip(tgt_tensor, tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            tgt_dict[(text, img_path)] = tgt_t

        eval_data = load_dataset(
            data_args.dataset_name,
            subset,
            split=data_args.dataset_split,
        )
        acc = 0
        all_pred = []
        for row in eval_data:
            qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
            tgt_t, all_candidates = [], []
            if row["tgt_text"] == "":
                row["tgt_text"] = ["" for _ in range(len(row["tgt_img_path"]))]
            for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                tgt_t.append(tgt_dict[tt])
                all_candidates.append(tt)
            try:
                tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            except:
                import ipdb; ipdb.set_trace()
            scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
            if pred == 0:
                acc += 1
            all_pred.append(all_candidates[pred])
            with open(os.path.join(data_args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
                for item in all_pred:
                    f.write(f"{item}\n")
        accuracy = acc / len(eval_data) * 100
        results[subset] = accuracy
        print(f"\033[91m{subset} accuracy: {acc/len(eval_data)}\033[0m")
    save_results(results, model_args, data_args, training_args)
    print_results(results)


if __name__ == "__main__":
    main()
