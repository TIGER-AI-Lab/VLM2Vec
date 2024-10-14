from src.model import MMEBModel
from src.arguments import ModelArguments
import torch
from transformers import HfArgumentParser, AutoProcessor
from PIL import Image
import numpy as np

model_args = ModelArguments(
    model_name='microsoft/Phi-3.5-vision-instruct', 
    pooling='last',
    normalize=True,
    lora=True,
    checkpoint_path='TIGER-Lab/VLM2Vec-LoRA')

model = MMEBModel.load(model_args)
model.eval()
model = model.to('cuda', dtype=torch.bfloat16)

processor = AutoProcessor.from_pretrained(
    model_args.model_name,
    trust_remote_code=True,
    num_crops=4,
)

inputs = processor(
    '<|image_1|> Represent the given image with the following question: What is in the image',
    [Image.open('figures/example.jpg')])
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

# Compute the similarity;
string = 'A cat and a dog'
inputs = processor(string, None, return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))

string = 'A cat and a tiger'
inputs = processor(string, None, return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))

string = 'A pig'
inputs = processor(string, None, return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))

string = 'a flight'
inputs = processor(string, None, return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))