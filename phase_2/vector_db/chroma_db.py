import chromadb
from chromadb import HttpClient

class ChromaDB:
    def __init__(self, text_collection="wiki_text", image_collection="wiki_image"):
        self.client = HttpClient(host="localhost", port=8000)
        # self.client = chromadb.PersistentClient(path="chroma_storage")

        for name in [text_collection, image_collection]:
            try:
                self.client.delete_collection(name=name)
            except Exception:
                pass

        self.text_collection  = self.client.create_collection(name=text_collection)
        self.image_collection = self.client.create_collection(name=image_collection)

    def add_text(self, ids, embeddings, documents, batch_size=500):
        documents = [str(d) if not isinstance(d, str) else d for d in documents]
        for i in range(0, len(ids), batch_size):
            self.text_collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size]
            )

    def add_image(self, ids, embeddings, documents, metadatas, batch_size=500):
        documents = [str(d) if not isinstance(d, str) else d for d in documents]
        for i in range(0, len(ids), batch_size):
            self.image_collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

    def query(self, query_embedding, k=3, modality="text"):
        collection = self.text_collection if modality == "text" else self.image_collection
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )



# import chromadb

# class ChromaDB:
#     def __init__(self, collection_name="wiki"):
#         self.client = chromadb.PersistentClient(path="chroma_storage")
        
#         try:
#             self.client.delete_collection(name=collection_name)
#         except Exception:
#             pass
        
#         self.collection = self.client.create_collection(name=collection_name)

#     def add(self, ids, embeddings, documents, batch_size=500):
#         # Coerce documents to strings (image_docs are dicts)
#         documents = [str(d) if not isinstance(d, str) else d for d in documents]
        
#         for i in range(0, len(ids), batch_size):
#             self.collection.add(
#                 ids=ids[i:i+batch_size],
#                 embeddings=embeddings[i:i+batch_size],
#                 documents=documents[i:i+batch_size]
#             )

#     def query(self, query_embedding, k=3):
#         return self.collection.query(
#             query_embeddings=[query_embedding], 
#             n_results=k
#         )