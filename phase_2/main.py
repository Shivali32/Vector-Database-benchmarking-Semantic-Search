# from data_loader import load_documents, chunk_text
from data_loader import load_documents, load_wit_images, chunk_text, os
from embedder import Embedder, np
from model_loader import load_model
from vector_db.chroma_db import ChromaDB
from vector_db.qdrant_db import QdrantDB
from vector_db.milvus_db import MilvusDB
from query_loader import load_queries
from query_engine import run_queries
from display import display_summary

def main(db_type="chroma", index_type="HNSW"):

    text_data_path = "wiki_dataset"
    image_metadata_path = "wit_metadata/wit_subset_metadata.json"
    query_path = "queries.json"

    text_docs = load_documents(text_data_path)

    text_chunks = []
    for doc in text_docs:
        text_chunks.extend(chunk_text(doc))
    # text_chunks = text_chunks[:1000]

    image_docs = load_wit_images(image_metadata_path)
    # image_docs = image_docs[:100] 

    ##  ---- SENTENCE TRANSFORMER -----
    # embedder = Embedder()
    # all_embeddings = embedder.embed_documents(text_chunks)
    # all_ids = [str(i) for i in range(len(text_chunks))]
    
    ## ---- CLIP -----
    model, processor, device = load_model()
    embedder = Embedder(model, processor, device)


    # ---- TEXT EMBEDDINGS ----
    if os.path.exists("text_emb.npy"):
        print("Loading cached text embeddings...")
        text_embeddings = np.load("embeddings/text_emb.npy")
    else:
        print("Computing text embeddings...")
        text_embeddings = embedder.embed_documents(text_chunks)
        np.save("embeddings/text_emb.npy", text_embeddings)

    text_ids = [f"text_{i}" for i in range(len(text_chunks))]


    # ---- IMAGE EMBEDDINGS ----
    if os.path.exists("image_emb.npy"):
        print("Loading cached image embeddings...")
        image_embeddings = np.load("embeddings/image_emb.npy")
    else:
        print("Computing image embeddings...")
        image_embeddings = embedder.embed_images(image_docs)
        np.save("embeddings/image_emb.npy", image_embeddings)

    image_ids = [doc["id"] for doc in image_docs]

    # all_embeddings = text_embeddings + image_embeddings
    all_embeddings = np.vstack([text_embeddings, image_embeddings])
    all_ids = text_ids + image_ids
    all_payloads = text_chunks + image_docs

    dim = len(all_embeddings[0])

    if db_type == "chroma":
        db = ChromaDB()
        db.collection.delete() 
    elif db_type == "qdrant":
        db = QdrantDB(dim=dim)
    elif db_type == "milvus":
        # db.drop()
        db = MilvusDB(dim=dim, index_type=index_type)

    ## ---- For text only ---
    # db.add(all_ids, all_embeddings, text_chunks)

    ## ---- For text + images ---
    db.add(all_ids, all_embeddings, all_payloads)

    queries = load_queries(query_path)
    results = run_queries(db, embedder, queries, db_type)

    display_summary(results)

if __name__ == "__main__":
    

    # for db in ["chroma", "qdrant", "milvus"]:
    for db in ["chroma", "qdrant"]:
        print(f"Running benchmark for: {db}")
        main(db, index_type="HNSW")

    # db = "milvus"
    # print(f"Running benchmark for: {db}")
    # main(db, index_type="DISKANN")