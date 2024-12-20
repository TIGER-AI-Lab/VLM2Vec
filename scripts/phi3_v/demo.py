from src.model import MMEBModel
from src.arguments import ModelArguments
import torch
from transformers import HfArgumentParser, AutoProcessor
from PIL import Image
import numpy as np

model_args = ModelArguments(
    model_name='TIGER-Lab/VLM2Vec-Full',
    pooling='last',
    normalize=True)

model = MMEBModel.load(model_args)
model.eval()
model = model.to('cuda', dtype=torch.bfloat16)

processor = AutoProcessor.from_pretrained(
    model_args.model_name,
    trust_remote_code=True,
    num_crops=4,
)

# Image + Text -> Text
inputs = processor('<|image_1|> Represent the given image with the following question: What is in the image', [Image.open(
    '../../figures/example.jpg')])
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = 'A cat and a dog'
inputs = processor(string)
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a dog = tensor([[0.2969]], device='cuda:0', dtype=torch.bfloat16)

string = 'A cat and a tiger'
inputs = processor(string)
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a tiger = tensor([[0.2080]], device='cuda:0', dtype=torch.bfloat16)

# Text -> Image
inputs = processor('Find me an everyday image that matches the given caption: A cat and a dog.',)
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = '<|image_1|> Represent the given image.'
inputs = processor(string, [Image.open('../../figures/example.jpg')])
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## <|image_1|> Represent the given image. = tensor([[0.3105]], device='cuda:0', dtype=torch.bfloat16)

inputs = processor('Find me an everyday image that matches the given caption: A cat and a tiger.',)
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = '<|image_1|> Represent the given image.'
inputs = processor(string, [Image.open('../../figures/example.jpg')])
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## <|image_1|> Represent the given image. = tensor([[0.2158]], device='cuda:0', dtype=torch.bfloat16)