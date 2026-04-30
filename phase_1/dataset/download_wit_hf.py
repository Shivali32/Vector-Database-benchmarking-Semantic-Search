from datasets import load_dataset
import json
import os
import requests
from tqdm import tqdm

OUTPUT_DIR = "wit_images_25k"
META_DIR = "wit_metadata"
META_FILE = os.path.join(META_DIR, "wit_metadata_25k.json")

MAX_IMAGES = 25000
SAVE_EVERY = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

metadata = []

# Resume if metadata file already exists
if os.path.exists(META_FILE):
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Resuming from {len(metadata)} downloaded images")

print("Loading WIT dataset in streaming mode...")

dataset = load_dataset(
    "wikimedia/wit_base",
    split="train",
    streaming=True
)

pbar = tqdm(total=MAX_IMAGES, initial=len(metadata), desc="Downloading images")

for item in dataset:
    if len(metadata) >= MAX_IMAGES:
        break

    image_url = item.get("image_url")
    caption = (
        item.get("caption_reference_description")
        or item.get("caption_attribution_description")
    )
    page_title = item.get("page_title")

    if not image_url or not caption:
        continue

    image_id = f"img_{len(metadata):05d}"
    image_path = os.path.join(OUTPUT_DIR, f"{image_id}.jpg")

    # Skip if already exists
    if os.path.exists(image_path):
        continue

    try:
        response = requests.get(image_url, timeout=10)

        if response.status_code != 200:
            continue

        with open(image_path, "wb") as f:
            f.write(response.content)

        metadata.append({
            "image_id": image_id,
            "page_title": page_title,
            "caption": caption,
            "image_path": image_path,
            "image_url": image_url,
            "type": "image",
            "source": "wit"
        })

        pbar.update(1)

        if len(metadata) % SAVE_EVERY == 0:
            with open(META_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

    except Exception:
        continue

pbar.close()

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Downloaded {len(metadata)} images")
print(f"Saved metadata to {META_FILE}")





# import os
# import json
# import requests
# from datasets import load_dataset
# from tqdm import tqdm
# from PIL import Image
# from io import BytesIO
# from itertools import islice

# NUM_SAMPLES = 100000
# IMAGE_DIR = "wit_images_100k"
# METADATA_DIR = "wit_metadata_100k"
# METADATA_FILE = os.path.join(METADATA_DIR, "wit_metadata.json")

# os.makedirs(IMAGE_DIR, exist_ok=True)
# os.makedirs(METADATA_DIR, exist_ok=True)

# print("Loading WIT dataset...")
# dataset = load_dataset("wikimedia/wit_base", split="train", streaming=True)

# subset = list(islice(dataset, NUM_SAMPLES))

# print(f"Collected {len(subset)} samples")

# metadata_list = []

# for idx, item in enumerate(tqdm(subset)):

#     img = item.get("image")
#     caption = item.get("caption_attribution_description")
#     page_title = item.get("page_title")

#     if img is None:
#         continue

#     image_id = f"img_{idx:04d}"
#     image_filename = f"{image_id}.jpg"
#     image_path = os.path.join(IMAGE_DIR, image_filename)

#     try:
#         img = img.convert("RGB")
#         img = img.resize((256, 256))
#         img.save(image_path)

#         metadata = {
#             "image_id": image_id,
#             "page_title": page_title,
#             "caption": caption,
#             "local_path": image_path,
#             "type": "image",
#             "source": "wit"
#         }

#         metadata_list.append(metadata)

#     except Exception as e:
#         print(f"Skipping due to error: {e}")
#         continue


# with open(METADATA_FILE, "w", encoding="utf-8") as f:
#     json.dump(metadata_list, f, indent=4)

# print("Done! Images and metadata saved successfully.")