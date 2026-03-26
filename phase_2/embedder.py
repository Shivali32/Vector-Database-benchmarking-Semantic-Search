
### CLIP VERSION
from PIL import Image
import torch
import numpy as np


class Embedder:

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def embed_documents(self, texts):

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()

    def embed_images(self, image_docs, batch_size=32):
        all_embeddings = []

        for i in range(0, len(image_docs), batch_size):
            batch = image_docs[i:i+batch_size]

            images = [
                Image.open(doc["image_path"]).convert("RGB")
                for doc in batch
            ]

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(image_features.cpu())

        return torch.cat(all_embeddings).numpy()

        # images = [
        #     Image.open(doc["image_path"]).convert("RGB")
        #     for doc in image_docs
        # ]

        # inputs = self.processor(
        #     images=images,
        #     return_tensors="pt"
        # ).to(self.device)

        # with torch.no_grad():
        #     image_features = self.model.get_image_features(**inputs)

        # return image_features.cpu().numpy()
    



### SENTENCE TRANSFORMER VERSION
# from sentence_transformers import SentenceTransformer

# class Embedder:
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)

#     def embed_documents(self, texts):
#         return self.model.encode(texts, show_progress_bar=True)
