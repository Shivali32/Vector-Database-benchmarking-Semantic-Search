from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantDB:
    def __init__(self, collection_name="wiki", dim=384):
        self.client = QdrantClient(path="qdrant_storage")
        self.collection_name = collection_name

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def add(self, ids, embeddings, documents):
        points = [
            PointStruct(id=int(i), vector=embedding, payload={"text": doc})
            for i, embedding, doc in zip(ids, embeddings, documents)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, query_embedding, k=3):
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,            
            # with_payload=True
        )
