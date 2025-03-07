from src.model import MMEBModel
from src.arguments import ModelArguments
from src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens
from PIL import Image
import torch

model_args = ModelArguments(
    model_name='Qwen/Qwen2-VL-2B-Instruct',
    checkpoint_path='TIGER-Lab/VLM2Vec-Qwen2VL-2B',
    pooling='last',
    normalize=True,
    model_backbone='qwen2_vl',
    lora=True
)

processor = load_processor(model_args)
model = MMEBModel.load(model_args)
model = model.to('cuda', dtype=torch.bfloat16)
model.eval()

# Image + Text -> Text
inputs = processor(text=f'{vlm_image_tokens[QWEN2_VL]} Represent the given image with the following question: What is in the image',
                   images=Image.open('figures/example.jpg'),
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
inputs['image_grid_thw'] = inputs['image_grid_thw'].unsqueeze(0)
qry_output = model(qry=inputs)["qry_reps"]

string = 'A cat and a dog'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a dog = tensor([[0.2500]], device='cuda:0', dtype=torch.bfloat16)

string = 'A cat and a tiger'
inputs = processor(text=string,
                   images=None,
                   return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
## A cat and a tiger = tensor([[0.1865]], device='cuda:0', dtype=torch.bfloat16)
