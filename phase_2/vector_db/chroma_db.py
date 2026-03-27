import chromadb

class ChromaDB:
    def __init__(self, collection_name="wiki"):
        self.client = chromadb.PersistentClient(path="chroma_storage")
        
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        
        self.collection = self.client.create_collection(name=collection_name)

    def add(self, ids, embeddings, documents, batch_size=500):
        # Coerce documents to strings (image_docs are dicts)
        documents = [str(d) if not isinstance(d, str) else d for d in documents]
        
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size]
            )

    def query(self, query_embedding, k=3):
        return self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=k
        )