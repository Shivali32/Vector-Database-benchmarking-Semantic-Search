import hashlib
import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)


class MilvusDB:
    def __init__(self, collection_name="wiki", dim=384, index_type="HNSW"):
        self.text_collection_name = "text_collection"
        self.image_collection_name = "image_collection"
        self.dim = dim
        self.index_type = index_type.upper()
        
        connections.connect(alias="default", host="localhost", port="19530")

        if utility.has_collection("text_collection"):
            utility.drop_collection("text_collection")

        if utility.has_collection("image_collection"):
            utility.drop_collection("image_collection")

        self.text_collection = self._create_collection("text_collection", 384)
        self.image_collection = self._create_collection("image_collection", 512)

    def _create_collection(self, name, dim):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="original_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=10)
        ]

        schema = CollectionSchema(fields)
        collection = Collection(name, schema)

        index_params = self._get_index_params()

        collection.create_index(
            field_name="vector",
            index_params=self._get_index_params(),
            # index_params=index_params
        )

        collection.load()
        return collection

    def _get_index_params(self):
        if self.index_type == "HNSW":
            return {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }

        elif self.index_type in ["IVF", "IVF_FLAT"]:
            return {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {
                    "nlist": 128
                }
            }

        elif self.index_type == "DISKANN":
            return {
                "metric_type": "COSINE",
                "index_type": "DISKANN",
                "params": {
                    "search_list": 100
                }
            }

        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")    

    def _string_to_int_id(self, string_id):
        """
        Convert string ID to integer using SHA256 (no collisions).
        Takes first 8 bytes of hash as 64-bit integer.
        """
        hash_bytes = hashlib.sha256(str(string_id).encode()).digest()
        return int.from_bytes(hash_bytes[:8], 'big') % (2**63 - 1)

    # Then replace the list comprehension:

    def add(self, ids, embeddings, documents):
        original_ids = [str(i) for i in ids]
        ids = [self._string_to_int_id(i) for i in ids]
        # ids = [int(abs(hash(str(i))) % (2**31)) for i in ids]
        
        # documents = [
        #     str(doc) if doc is not None else ""
        #     for doc in documents
        # ]
        
        # types = []
        # for doc in documents:
        #     if isinstance(doc, str):
        #         types.append("text")
        #     else:
        #         types.append("image")

        types = ["image" if isinstance(doc, dict) else "text" for doc in documents]
        documents = [str(doc) if doc is not None else "" for doc in documents]


        embeddings = self.normalize(embeddings)

        self.collection.insert([
            ids,
            embeddings,
            documents,
            original_ids,
            types
        ])

        self.collection.flush()

    def _get_search_params(self):
        if self.index_type == "HNSW":
            return {
                "metric_type": "COSINE",
                "params": {
                    "ef": 64
                }
            }

        elif self.index_type in ["IVF", "IVF_FLAT"]:
            return {
                "metric_type": "COSINE",
                "params": {
                    "nprobe": 10
                }
            }

        elif self.index_type == "DISKANN":
            return {
                "metric_type": "COSINE",
                "params": {
                    "search_list": 100
                }
            }

    def query(self, query_embedding, k=3, modality="text"):
        collection = self.text_collection if modality == "text" else self.image_collection
        
        search_params = self._get_search_params()
        query_embedding = self.normalize([query_embedding])[0]

        results = collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["text", "original_id", "type", "vector"]
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "text": hit.entity.get("text"),
                "original_id": hit.entity.get("original_id"),
                "type": hit.entity.get("type"),
                "vector": hit.entity.get("vector"),
                "score": hit.distance
            })

        return hits

    def drop(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

    def count(self):
        return self.collection.num_entities
    

    def normalize(self, vecs):
        vecs = np.array(vecs)
        return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).tolist()


    def add_text(self, ids, embeddings, documents):
        self._insert(self.text_collection, ids, embeddings, documents, "text")

    def add_image(self, ids, embeddings, documents, metadatas=None):
        if metadatas:
            clean_docs = []
            original_ids = []

            for idx, doc in enumerate(documents):
                meta = metadatas[idx]

                # store ONLY clean caption text
                clean_docs.append(meta.get("caption", doc))

                # keep original_id separately
                original_ids.append(str(ids[idx]))

            self._insert(
                self.image_collection,
                ids,
                embeddings,
                clean_docs,
                "image",
                original_ids=original_ids
            )
        else:
            self._insert(self.image_collection, ids, embeddings, documents, "image")

    def _insert(self, collection, ids, embeddings, documents, dtype, original_ids=None):
        if original_ids is None:
            original_ids = [str(i) for i in ids]

        ids = [int(abs(hash(str(i))) % (2**31)) for i in ids]

        documents = [str(doc) if doc is not None else "" for doc in documents]
        types = [dtype] * len(documents)

        embeddings = self.normalize(embeddings)

        collection.insert([
            ids,
            embeddings,
            documents,
            original_ids,
            types
        ])

        collection.flush()