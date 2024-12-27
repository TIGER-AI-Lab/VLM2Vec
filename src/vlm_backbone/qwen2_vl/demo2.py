import torch
from PIL import Image

from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

model_name = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
model.padding_side = "right"
image_processor = Qwen2VLImageProcessor.from_pretrained(model_name)
tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
processor = Qwen2VLProcessor.from_pretrained(
    model_name,
    # image_processor=image_processor,
    # tokenizer=tokenizer,
    # num_crops=model_args.num_crops
)

# Your inputs
size = (140, 140)
images = [
    Image.new("RGB", size, color="white"),
    # Image.new("RGB", size, color="black"),
]
queries = [
    "<|image_pad|>Is attention really all you need?",
    # "<|image_pad|>What is the amount of bananas farmed in Salvador?",
]

# Process the inputs
batch = processor(images=images, text=queries[0], return_tensors="pt")
print(batch["input_ids"])
print(batch["pixel_values"])
print(batch["image_grid_thw"])
# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch)
    query_embeddings = model(**batch)

print(image_embeddings.keys())
print()
print(query_embeddings.keys())