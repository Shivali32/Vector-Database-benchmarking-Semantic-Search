from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantDB:
    def __init__(self, collection_name="wiki", dim=384):
        self.client = QdrantClient(path="qdrant_storage")
        self.collection_name = collection_name
        
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

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

    def query(self, query_embedding, k=3):
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            with_vectors=True,
            with_payload=True
        )
        return result.points
