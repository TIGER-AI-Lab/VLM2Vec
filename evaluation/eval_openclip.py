import open_clip
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor, AutoTokenizer, CLIPModel
from src.dataset import EvalDataset
from src.collator import EvalCollator, BLIP2Collator, CLIPCollator, OpenCLIPCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.eval_utils import get_pred, save_results, print_results

t2i_tasks = [
    "CIRR", "NIGHTS", "EDIS", "MSCOCO_t2i","VisDial","VisualNews_t2i","WebQA", "Wiki-SS-NQ", "OVEN", # retrieval
    ]
i2t_tasks = [
    "MSCOCO_i2t","VisualNews_i2t", # retrieval
    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211" # classification
    ]


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    model, processor = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')

    embedding_type = data_args.embedding_type
    eval_collator = OpenCLIPCollator(
        data_args=data_args,
        vis_processors=processor,
        txt_processors=tokenizer
    )
    model.eval()
    model = model.to(training_args.device)

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
                batch = {key: value.to(training_args.device) for key, value in batch.items() if type(value) is not list}
                image_features, text_features = None, None
                if "pixel_values" in batch:
                    image_features = model.encode_image(batch["pixel_values"])
                if "input_ids" in batch:
                    text_features = model.encode_text(batch["input_ids"])
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
                        try:
                            features = image_features + text_features
                        except:
                            import ipdb; ipdb.set_trace()
                encoded_tensor.append(features.cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_qry_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                batch = {key: value.to(training_args.device) for key, value in batch.items() if type(value) is not list}
                image_features, text_features = None, None
                if "pixel_values" in batch:
                    image_features = model.encode_image(batch["pixel_values"])
                if "input_ids" in batch:
                    text_features = model.encode_text(batch["input_ids"])
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
            encoded_tensor = np.concatenate(encoded_tensor)
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
