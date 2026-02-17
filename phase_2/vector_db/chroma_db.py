import chromadb
from chromadb.config import Settings

class ChromaDB:
    def __init__(self, collection_name="wiki"):
        self.client = chromadb.Client(
            Settings(persist_directory="chroma_storage", is_persistent=True)
        )
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, ids, embeddings, documents, batch_size=500):
        # self.collection.add(ids=ids, embeddings=embeddings, documents=documents)        

        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size]
            )

    def query(self, query_embedding, k=3):
        return self.collection.query(query_embeddings=[query_embedding], n_results=k)
