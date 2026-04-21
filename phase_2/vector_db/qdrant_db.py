from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantDB:
    def __init__(self, collection_name="wiki", dim=384):
        self.client = QdrantClient(path="qdrant_storage")
        self.text_collection = "text_collection"
        self.image_collection = "image_collection"
        
        if self.client.collection_exists(self.text_collection):
            self.client.delete_collection(self.text_collection)

        if self.client.collection_exists(self.image_collection):
            self.client.delete_collection(self.image_collection)           

        self.client.create_collection(
            collection_name="text_collection",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        self.client.create_collection(
            collection_name="image_collection",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
        # self.client.create_collection(
        #     collection_name=collection_name,
        #     vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        # )

    def _to_int_id(self, id_):
        # If already int, use it. If string, hash it to a valid int.
        try:
            return int(id_)
        except (ValueError, TypeError):
            return abs(hash(str(id_))) % (2**31)

    def add(self, ids, embeddings, documents):
        points = [
            PointStruct(
                id=self._to_int_id(i),
                vector=embedding,
                payload={"text": doc, "original_id": str(i)}
            )
            for i, embedding, doc in zip(ids, embeddings, documents)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, query_embedding, k=3, modality="text"):
        collection = "text_collection" if modality == "text" else "image_collection"

        result = self.client.query_points(
            collection_name=collection,
            query=query_embedding,
            limit=k,
            with_vectors=True,
            with_payload=True
        )
        return result.points

    def add_text(self, ids, embeddings, documents):
        self.client.upsert(
            collection_name="text_collection",
            points=[
                PointStruct(
                    id=self._to_int_id(i),
                    vector=embedding,
                    payload={
                        "text": doc,
                        "original_id": str(i)
                    }
                )
                for i, embedding, doc in zip(ids, embeddings, documents)
            ]
        )

    def add_image(self, ids, embeddings, documents, metadatas=None):
        self.client.upsert(
            collection_name="image_collection",
            points=[
                PointStruct(
                    id=self._to_int_id(i),
                    vector=embedding,
                    payload={
                        "text": doc,
                        "original_id": str(i),
                        **(metadatas[idx] if metadatas else {})
                    }
                )
                for idx, (i, embedding, doc) in enumerate(zip(ids, embeddings, documents))
            ]
        )        