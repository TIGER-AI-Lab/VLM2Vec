from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
from tqdm import tqdm
from archive.instruction_dict import INSTRUCTION_DICT
import random


# all_subsets_only_inst = ["ImageNet_1K", "HatefulMemes", "SUN397", "VOC2007"]

# all_subsets_has_text = ["N24News"]
# all_subsets_has_text = ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W"]
# all_subsets_has_text = ["MSCOCO"]

all_subsets_only_inst = ["NIGHTS", "MSCOCO_i2t", "VisualNews_i2t"]

all_subsets_has_text = ["VisDial", "CIRR", "WebQA", "MSCOCO_t2i", "VisualNews_t2i"]

login(token="")

def update_instruction(example, subset_name):
    """
    Fix the image special token issue.
    """
    example["qry"] = random.choice(INSTRUCTION_DICT[subset_name])
    return example


def update_instruction_has_text(example, subset_name):
    """
    Fix the image special token issue.
    """
    original_inst = INSTRUCTION_DICT[subset_name][0]
    text = example["qry"][len(original_inst):]
    example["qry"] = random.choice(INSTRUCTION_DICT[subset_name]) + text

    return example

for subset_name in tqdm(all_subsets_only_inst):
    print(f"Processing subset {subset_name}")
    subset = load_dataset("ziyjiang/mmeb_final", subset_name)
    if "train" in subset:
        original_split = subset["train"]
    else:
        original_split = subset["original"]
    subset = DatasetDict({"original": original_split})

    diverse_instruction_split = original_split.map(
        lambda example: update_instruction(example, subset_name))
    subset["diverse_instruction"] = diverse_instruction_split

    subset.push_to_hub("TIGER-Lab/MMEB-train", subset_name)


for subset_name in tqdm(all_subsets_has_text):
    print(f"Processing subset {subset_name}")
    subset = load_dataset("ziyjiang/mmeb_final", subset_name)
    if "train" in subset:
        original_split = subset["train"]
    else:
        original_split = subset["original"]
    subset = DatasetDict({"original": original_split})

    diverse_instruction_split = original_split.map(
        lambda example: update_instruction_has_text(example, subset_name))
    subset["diverse_instruction"] = diverse_instruction_split

    subset.push_to_hub("TIGER-Lab/MMEB-train", subset_name)
