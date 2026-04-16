from PIL import Image
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, clip_model, clip_processor, device):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts, batch_size=32, save_path="embeddings/text_embeddings.npy"):

        if save_path and self._exists(save_path):
            return self._load_embeddings(save_path)

        print("Computing text embeddings...")
        
        if isinstance(texts[0], dict):
            texts = [t["content"] for t in texts]

        embeddings = self.text_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        if save_path:
            self._save_embeddings(embeddings, save_path)

        return embeddings        

        # all_embeddings = []

        # for i in range(0, len(texts), batch_size):
        #     batch = texts[i:i + batch_size]

        #     inputs = self.processor(
        #         text=batch,
        #         return_tensors="pt",
        #         padding=True,
        #         truncation=True
        #     ).to(self.device)

        #     with torch.no_grad():
        #         text_out = self.model.text_model(
        #             input_ids=inputs["input_ids"],
        #             attention_mask=inputs["attention_mask"]
        #         )
        #         text_features = self.model.text_projection(text_out.pooler_output)

        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #     all_embeddings.append(text_features.cpu())

        # result = torch.cat(all_embeddings).numpy()

        # assert result.ndim == 2
        # assert result.shape[1] == 512

        # if save_path:
        #     self._save_embeddings(result, save_path)
        # return result

    # def embed_documents(self, texts, save_path="embeddings/text_embeddings.npy"):
        
        # if self._exists(save_path):
        #     return self._load_embeddings(save_path)
        
        # print("Computing text embeddings...")
        # inputs = self.processor(
        #     text=texts,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True
        # ).to(self.device)

        # with torch.no_grad():
        #     text_out = self.model.text_model(
        #         input_ids=inputs["input_ids"],
        #         attention_mask=inputs["attention_mask"]
        #     )
        #     text_features = self.model.text_projection(text_out.pooler_output)

        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # result = text_features.cpu().numpy()

        # assert result.ndim == 2, f"embed_documents: expected 2D, got {result.shape}"
        # assert result.shape[1] == 512, f"embed_documents: expected 512-dim, got {result.shape}"
        
        # self._save_embeddings(result, save_path)
        # return result
        

    def embed_images(self, image_docs, batch_size=32, save_path="embeddings/image_embeddings.npy"):

        if save_path and self._exists(save_path):
            return self._load_embeddings(save_path)
        
        print("Computing image embeddings...")
        
        all_embeddings = []

        for i in range(0, len(image_docs), batch_size):
            batch = image_docs[i:i + batch_size]
            images = [Image.open(doc["image_path"]).convert("RGB") for doc in batch]

            inputs = self.clip_processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                image_out = self.clip_model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )
                image_features = self.clip_model.visual_projection(image_out.pooler_output)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu())

        result = torch.cat(all_embeddings).numpy()
        assert result.ndim == 2, f"embed_images: expected 2D, got {result.shape}"
        assert result.shape[1] == 512, f"embed_images: expected 512-dim, got {result.shape}"

        self._save_embeddings(result, save_path)
        return result

    def embed_queries(self, queries, batch_size=32, save_path="embeddings/query_embeddings.npy", use_cache=True):

        if use_cache and save_path and self._exists(save_path):
            return self._load_embeddings(save_path)

        if isinstance(queries[0], dict):
            queries = [q["content"] for q in queries]


        embeddings = self.text_model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        if save_path:
            self._save_embeddings(embeddings, save_path)

        return embeddings

        # all_embeddings = []

        # for i in range(0, len(queries), batch_size):
        #     batch = queries[i:i + batch_size]

        #     inputs = self.processor(
        #         text=batch,
        #         return_tensors="pt",
        #         padding=True,
        #         truncation=True
        #     ).to(self.device)

        #     with torch.no_grad():
        #         text_out = self.model.text_model(
        #             input_ids=inputs["input_ids"],
        #             attention_mask=inputs["attention_mask"]
        #         )
        #         text_features = self.model.text_projection(text_out.pooler_output)

        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #     all_embeddings.append(text_features.cpu())

        # result = torch.cat(all_embeddings).numpy()

        # if save_path:
        #     self._save_embeddings(result, save_path)
        # return result


    # def embed_queries(self, queries, save_path="embeddings/query_embeddings.npy"):

    #     if self._exists(save_path):
    #         return self._load_embeddings(save_path)

    #     inputs = self.processor(
    #         text=queries,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True
    #     ).to(self.device)

    #     with torch.no_grad():
    #         text_out = self.model.text_model(
    #             input_ids=inputs["input_ids"],
    #             attention_mask=inputs["attention_mask"]
    #         )
    #         text_features = self.model.text_projection(text_out.pooler_output)

    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #     result = text_features.cpu().numpy()

    #     self._save_embeddings(result, save_path)
    #     return result
    #     # return self.embed_documents([query_text])[0]


    def embed_image_query(self, query):
        inputs = self.clip_processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        with torch.no_grad():
            text_out = self.clip_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            text_features = self.clip_model.text_projection(text_out.pooler_output)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

    def _save_embeddings(self, embeddings, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, embeddings)
        print(f"Saved embeddings → {path}")

    def _load_embeddings(self, path):
        embeddings = np.load(path)
        print(f"Loaded embeddings → {path}")
        return embeddings

    def _exists(self, path):
        return os.path.exists(path)
    


# ### SENTENCE TRANSFORMER VERSION
# # from sentence_transformers import SentenceTransformer

# # class Embedder:
# #     def __init__(self, model_name="all-MiniLM-L6-v2"):
# #         self.model = SentenceTransformer(model_name)

# #     def embed_documents(self, texts):
# #         return self.model.encode(texts, show_progress_bar=True)

