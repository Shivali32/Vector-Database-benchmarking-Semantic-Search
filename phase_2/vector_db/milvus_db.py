from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

class MilvusDB:
    def __init__(self, collection_name="wiki", dim=384):
        self.collection_name = collection_name
        self.dim = dim

        connections.connect(alias="default", host="localhost", port="19530")
        # connections.connect(alias="default", uri="milvus_lite.db")

        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
        else:
            self.collection = self._create_collection()

    def _create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
        ]

        schema = CollectionSchema(fields)
        collection = Collection(self.collection_name, schema)

        index_params = {
            "metric_type": "COSINE",
            # "index_type": "IVF_FLAT",
            "index_type": "HNSW",
            # "index_type": "DiskANN",
            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }

        # collection.create_index(field_name="vector", index_params=index_params)
        # collection.load()

        return collection

    def add(self, ids, embeddings, documents):
        ids = [int(i) for i in ids]

        self.collection.insert([
            ids,
            embeddings,
            documents
        ])

        self.collection.flush()

        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }

        self.collection.create_index(
            field_name="vector",
            index_params=index_params
        )

        self.collection.load()

    def query(self, query_embedding, k=3):
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }

        results = self.collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["text"]
        )

        return results[0]
