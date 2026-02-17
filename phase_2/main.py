from data_loader import load_documents, chunk_text
from embedder import Embedder
from vector_db.chroma_db import ChromaDB
from vector_db.qdrant_db import QdrantDB
from vector_db.milvus_db import MilvusDB
from query_loader import load_queries
from query_engine import run_queries
from display import display_summary

def main(db_type="chroma"):
    data_path = "wiki_dataset"
    query_path = "queries.json"

    documents = load_documents(data_path)

    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc))

    embedder = Embedder()
    embeddings = embedder.embed_documents(chunks)
    ids = [str(i) for i in range(len(chunks))]

    if db_type == "chroma":
        db = ChromaDB()
    elif db_type == "qdrant":
        db = QdrantDB(dim=len(embeddings[0]))
    elif db_type == "milvus":
        db = MilvusDB(dim=len(embeddings[0]))

    db.add(ids, embeddings, chunks)

    queries = load_queries(query_path)
    results = run_queries(db, embedder, queries, db_type)

    display_summary(results)

if __name__ == "__main__":
    

    # for db in ["chroma", "qdrant", "milvus"]:
    #     print(f"Running benchmark for: {db}")
    #     main(db)

    main("milvus")