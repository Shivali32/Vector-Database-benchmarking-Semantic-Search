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
            text_out = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            text_features = self.model.text_projection(text_out.pooler_output)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        result = text_features.cpu().numpy()
        assert result.ndim == 2, f"embed_documents: expected 2D, got {result.shape}"
        assert result.shape[1] == 512, f"embed_documents: expected 512-dim, got {result.shape}"
        return result  # (N, 512)

    def embed_images(self, image_docs, batch_size=32):
        all_embeddings = []

        for i in range(0, len(image_docs), batch_size):
            batch = image_docs[i:i + batch_size]
            images = [Image.open(doc["image_path"]).convert("RGB") for doc in batch]

            inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                image_out = self.model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )
                image_features = self.model.visual_projection(image_out.pooler_output)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu())

        result = torch.cat(all_embeddings).numpy()
        assert result.ndim == 2, f"embed_images: expected 2D, got {result.shape}"
        assert result.shape[1] == 512, f"embed_images: expected 512-dim, got {result.shape}"
        return result  # (N, 512)

    def embed_query(self, query_text):
        return self.embed_documents([query_text])[0]  # (512,)
    






# ### CLIP VERSION
# from PIL import Image
# import torch
# import numpy as np


# class Embedder:

#     def __init__(self, model, processor, device):
#         self.model = model
#         self.processor = processor
#         self.device = device

#     def embed_documents(self, texts):

#         inputs = self.processor(
#             text=texts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True
#         ).to(self.device)

#         with torch.no_grad():
#             text_features = self.model.get_text_features(**inputs)

#         if not isinstance(text_features, torch.Tensor):
#             text_features = text_features[0]

#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         return text_features.cpu().numpy()       


#     def embed_images(self, image_docs, batch_size=32):

#         all_embeddings = []
#         for i in range(0, len(image_docs), batch_size):
#             batch = image_docs[i:i+batch_size]
#             images = [Image.open(doc["image_path"]).convert("RGB") for doc in batch]

#             inputs = self.processor(images=images, return_tensors="pt").to(self.device)

#             with torch.no_grad():
#                 image_features = self.model.get_image_features(**inputs)

#             if not isinstance(image_features, torch.Tensor):
#                 image_features = image_features[0]

#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#             all_embeddings.append(image_features.cpu())

#         return torch.cat(all_embeddings).numpy()



# ### SENTENCE TRANSFORMER VERSION
# # from sentence_transformers import SentenceTransformer

# # class Embedder:
# #     def __init__(self, model_name="all-MiniLM-L6-v2"):
# #         self.model = SentenceTransformer(model_name)

# #     def embed_documents(self, texts):
# #         return self.model.encode(texts, show_progress_bar=True)

