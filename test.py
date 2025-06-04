
from src.model import MMEBModel
from src.arguments import ModelArguments
import torch
from transformers import HfArgumentParser, AutoProcessor
from PIL import Image
import numpy as np

from src.model_utils import Phi3V_process_fn


def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch


model_name = '/data/mohbat/models/VLM2Vec-Full/'
phi3_path = '/data/mohbat/models/phi/Phi-3.5-vision-instruct'
device = 'cuda:1'

model_args = ModelArguments(
    model_name=model_name,
    pooling='last',
    normalize=True,
    model_backbone='phi3_v',
    num_crops=4)

processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

model = MMEBModel.load(model_args)
model.eval()
model = model.to(device, dtype=torch.bfloat16)

text = '<|image_1|> Represent the given image with the following question: What is in the image'
images = [Image.open('figures/example.jpg')]
# Image + Text -> Text
inputs = Phi3V_process_fn({'text':[text], 'image':images}, processor, max_length=512)
batch = batch_to_device(inputs, device)

    
with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
    output = model(qry=batch)
    qry_output = output["qry_reps"]#.cpu().detach().float().numpy()
    print (qry_output.shape)
    
    
string = 'A cat and a dog'

inputs = Phi3V_process_fn({'text':[text]}, processor, max_length=512)
batch = batch_to_device(inputs, device)

with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
    output = model(tgt=batch)
    tgt_output = output["tgt_reps"]#.cpu().detach().float().numpy()
    print (tgt_output.shape)
    

print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a dog = tensor([[0.3008]], device='cuda:0', dtype=torch.b