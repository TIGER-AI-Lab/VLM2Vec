import random
from typing import List, Tuple
from itertools import islice
import datasets
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
from PIL import Image
import os


Phi_Image_token = "<|image_1|>"
Llava_Image_token = "<image>"
class TrainDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        train_data = []
        print(f"Loading {len(data_args.subset_name)} datasets: {data_args.subset_name}")
        for subset in data_args.subset_name:
            subset_data = load_dataset(
                self.data_args.dataset_name,
                subset,
                split=f"{self.data_args.dataset_split}[:{data_args.num_sample_per_subset}]",
            )
            train_data.append(subset_data)
        self.train_data = concatenate_datasets(train_data)

    def __len__(self):
        return len(self.train_data)

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone == "llava":
            return self._process_image(image, "low")
        else:
            return image

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        qry_text, qry_image_path, pos_text, pos_image_path = (
            self.train_data[item]["qry"], self.train_data[item]["qry_image_path"],
            self.train_data[item]["pos_text"], self.train_data[item]["pos_image_path"],
        )
        if self.model_args.model_backbone == "llava":
            # Update image token
            qry_text = qry_text.replace(Phi_Image_token, Llava_Image_token)
            pos_text = pos_text.replace(Phi_Image_token, Llava_Image_token)
        return (qry_text, self._get_image(qry_image_path),
                pos_text, self._get_image(pos_image_path))


class EvalDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args

        self.eval_data = load_dataset(
            self.data_args.dataset_name,
            subset,
            split=self.data_args.dataset_split,
        )
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        self.paired_dataset = datasets.Dataset.from_dict({
            "text": [pair["text"] for pair in self.paired_data],
            "img_path": [pair["img_path"] for pair in self.paired_data]
        })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.model_args.model_backbone == "llava":
            # Update llava image token
            text = text.replace(Phi_Image_token, Llava_Image_token)
        return text, self._get_image(img_path),

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone == "llava":
            return self._process_image(image, "low")
        else:
            return image

        return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        unique_pair = set()
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if row[text_field]:
                    unique_pair.add((row[text_field], row[img_path_field]))
                else:
                    if isinstance(row[img_path_field], List):
                        for img_path in row[img_path_field]:
                            unique_pair.add((row[text_field], img_path))
                    else:
                        unique_pair.add((row[text_field], row[img_path_field]))
            elif isinstance(row[text_field], List):
                assert isinstance(row[img_path_field], List) and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    unique_pair.add((text, img_path))

        paired_data = [{"text": text, "img_path": img_path} for text, img_path in unique_pair]
        return paired_data


class FlickrDataset(Dataset):
    def __init__(self, modality):
        self.raw_data = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval", split="test")
        if modality == "image":
            self.eval_data, self.image_names = self.get_image_data()
        else:
            self.eval_data, self.image_names = self.get_text_data()

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        return self.eval_data[idx]

    def get_image_data(self):
        eval_data, image_names = [], []
        # inst = "<|image_1|> Find an image caption describing the given image."
        inst = "<|image_1|> Represent the given image for image caption retrieval."
        for row in self.raw_data:
            eval_data.append((inst, row["image"]))
            image_names.append(row["filename"])
        return eval_data, image_names

    def get_text_data(self):
        eval_data, image_names = [], []
        inst = "Retrieve an image that matches the given caption: "
        for row in self.raw_data:
            for caption in row["caption"]:
                eval_data.append((caption, None))
                image_names.append(row["filename"])
        return eval_data, image_names
