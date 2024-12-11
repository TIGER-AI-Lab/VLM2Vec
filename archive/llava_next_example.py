from PIL import Image
import requests
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = model.cuda()
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image = None

inputs = processor(images=image, text=prompt, return_tensors="pt")
inputs = inputs.to(model.device)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(output)
pass
