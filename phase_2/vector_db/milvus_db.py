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
        self.collection_name = collection_name
        self.dim = dim
        self.index_type = index_type.upper()
        
        connections.connect(alias="default", host="localhost", port="19530")

        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
        else:
            self.collection = self._create_collection()

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="original_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=10)
        ]

        schema = CollectionSchema(fields)
        collection = Collection(self.collection_name, schema)

        index_params = self._get_index_params()

        collection.create_index(
            field_name="vector",
            index_params=index_params
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

    def add(self, ids, embeddings, documents):
        original_ids = [str(i) for i in ids]
        ids = [int(abs(hash(str(i))) % (2**31)) for i in ids]
        
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

    def query(self, query_embedding, k=3):
        search_params = self._get_search_params()

        query_embedding = self.normalize([query_embedding])[0]

        results = self.collection.search(
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
    