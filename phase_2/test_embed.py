# test_embed.py
from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
model.eval()

# Test text
texts = ["a dog", "a cat"]
text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_out = model.text_model(**text_inputs)
    text_feat = model.text_projection(text_out.pooler_output)
print(f"text: {text_feat.shape}")  # expect (2, 512)

# Test image - use any image you have
img = Image.new("RGB", (224, 224))  # blank image, no file needed
image_inputs = processor(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    image_out = model.vision_model(**image_inputs)
    image_feat = model.visual_projection(image_out.pooler_output)
print(f"image: {image_feat.shape}")  # expect (1, 512)