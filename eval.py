import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoTokenizer, AutoProcessor

from src.model import MMEBModel
from src.dataset import EvalDataset
from src.collator import EvalCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.eval_utils import get_pred


def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

    processor.tokenizer.padding_side = "right"
    model = MMEBModel.load(model_args, training_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    eval_collator = EvalCollator(
        data_args=data_args,
        processor=processor,
    )


    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(data_args.subset_name):
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
            except Exception as e:
                pass

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
                batch = {key: value.to(training_args.device) for key, value in batch.items()}
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output = model(qry=batch)
                encoded_tensor.append(output["qry_reps"].cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_qry_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                batch = {key: value.to(training_args.device) for key, value in batch.items()}
                output = model(tgt=batch)
                encoded_tensor.append(output["tgt_reps"].cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_tgt_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

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
        n_correct = 0
        all_pred = []
        for row in eval_data:
            qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
            tgt_t, all_candidates = [], []
            for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                tgt_t.append(tgt_dict[tt])
                all_candidates.append(tt)
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
            if pred == 0:
                n_correct += 1
            all_pred.append(all_candidates[pred])
        with open(os.path.join(data_args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            score_dict = {"acc": n_correct/len(eval_data), "num_correct": n_correct, "num_pred": len(eval_data)}
            json.dump(score_dict, f, indent=4)
        print(f"\033[91m{subset} accuracy: {n_correct/len(eval_data)}\033[0m")


if __name__ == "__main__":
    main()
