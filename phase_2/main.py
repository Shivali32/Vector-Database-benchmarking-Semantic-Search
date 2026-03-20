# from data_loader import load_documents, chunk_text
from data_loader import load_documents, load_wit_images, chunk_text
from embedder import Embedder
from model_loader import load_model
from vector_db.chroma_db import ChromaDB
from vector_db.qdrant_db import QdrantDB
from vector_db.milvus_db import MilvusDB
from query_loader import load_queries
from query_engine import run_queries
from display import display_summary

def main(db_type="chroma"):

    text_data_path = "wiki_dataset"
    image_metadata_path = "wit_metadata/wit_subset_metadata.json"
    query_path = "queries.json"

    text_docs = load_documents(text_data_path)

    text_chunks = []
    for doc in text_docs:
        text_chunks.extend(chunk_text(doc))

    image_docs = load_wit_images(image_metadata_path)

    embedder = Embedder()
    all_embeddings = embedder.embed_documents(text_chunks)
    all_ids = [str(i) for i in range(len(text_chunks))]
    
    
    # model, processor, device = load_model()
    # embedder = Embedder(model, processor, device)

    # text_embeddings = embedder.embed_documents(text_chunks)
    # text_ids = [f"text_{i}" for i in range(len(text_chunks))]

    # image_embeddings = embedder.embed_images(image_docs)
    # image_ids = [doc["id"] for doc in image_docs]

    # all_embeddings = text_embeddings + image_embeddings
    # all_ids = text_ids + image_ids
    # all_payloads = text_chunks + image_docs

    dim = len(all_embeddings[0])

    if db_type == "chroma":
        db = ChromaDB()
    elif db_type == "qdrant":
        db = QdrantDB(dim=dim)
    elif db_type == "milvus":
        db = MilvusDB(dim=dim)

    db.add(all_ids, all_embeddings, text_chunks)
    # db.add(all_ids, all_embeddings, all_payloads)

    queries = load_queries(query_path)
    results = run_queries(db, embedder, queries, db_type)

    display_summary(results)

if __name__ == "__main__":
    

    # for db in ["chroma", "qdrant", "milvus"]:
    # for db in ["chroma", "qdrant"]:
    #     print(f"Running benchmark for: {db}")
    #     main(db)

    db = "milvus"
    print(f"Running benchmark for: {db}")
    main(db)