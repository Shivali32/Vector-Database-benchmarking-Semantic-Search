import os
import json
import requests
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from itertools import islice

NUM_SAMPLES = 1000
IMAGE_DIR = "wit_images"
METADATA_DIR = "wit_metadata"
METADATA_FILE = os.path.join(METADATA_DIR, "wit_subset_metadata.json")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

print("Loading WIT dataset...")
dataset = load_dataset("wikimedia/wit_base", split="train", streaming=True)

subset = list(islice(dataset, NUM_SAMPLES))

print(f"Collected {len(subset)} samples")

metadata_list = []

for idx, item in enumerate(tqdm(subset)):

    img = item.get("image")
    caption = item.get("caption_attribution_description")
    page_title = item.get("page_title")

    if img is None:
        continue

    image_id = f"img_{idx:04d}"
    image_filename = f"{image_id}.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)

    try:
        img = img.convert("RGB")
        img = img.resize((256, 256))
        img.save(image_path)

        metadata = {
            "image_id": image_id,
            "page_title": page_title,
            "caption": caption,
            "local_path": image_path,
            "type": "image",
            "source": "wit"
        }

        metadata_list.append(metadata)

    except Exception as e:
        print(f"Skipping due to error: {e}")
        continue


with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, indent=4)

print("Done! Images and metadata saved successfully.")