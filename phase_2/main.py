from data_loader import load_documents, load_wit_images, chunk_text, os
from embedder import Embedder, np
from model_loader import load_model
from vector_db.chroma_db import ChromaDB
from vector_db.qdrant_db import QdrantDB
from vector_db.milvus_db import MilvusDB
from query_loader import load_queries
from query_engine import run_queries
from display import display_summary
from db_logger import init_db, log_result

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


TEXT_DATA_PATH   = "wiki_dataset"
IMAGE_META_PATH  = "wit_metadata/wit_subset_metadata.json"
QUERY_PATH       = "queries/queries_marco.json"
EMBED_DIM        = 512

# text_data_path = "wiki_sub"
# image_metadata_path = "wit_metadata/wit_meta_sub.json"


def load_text_chunks():
    text_docs = load_documents(TEXT_DATA_PATH)
    text_chunks = []
    for doc in text_docs:
        text_chunks.extend(chunk_text(doc))
    # text_chunks = text_chunks[:10]
    return text_chunks


def load_image_docs():
    image_docs = load_wit_images(IMAGE_META_PATH)
    # image_docs = image_docs[:10]
    return image_docs


def build_embedder():
    ##  ---- SENTENCE TRANSFORMER -----
    # embedder = Embedder()
    # return embedder

    ## ---- CLIP -----
    model, processor, device = load_model()
    embedder = Embedder(model, processor, device)
    return embedder


def get_text_embeddings(embedder, text_chunks):
    # if os.path.exists("embeddings/text_emb.npy"):
    #     print("Loading cached text embeddings...")
    #     text_embeddings = np.load("embeddings/text_emb.npy", allow_pickle=True)
    #     text_embeddings = np.squeeze(text_embeddings)          # (N,1,512) → (N,512)
    #     assert text_embeddings.ndim == 2, f"Expected 2D, got {text_embeddings.shape}"
    #     return text_embeddings.tolist()

    print("Computing text embeddings...")
    text_embeddings = embedder.embed_documents(text_chunks)

    # text_embeddings = np.array(text_embeddings)
    # text_embeddings = np.squeeze(text_embeddings)          # fix shape before saving

    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/text_emb.npy", text_embeddings)   # save already-squeezed

    return text_embeddings.tolist()


def get_image_embeddings(embedder, image_docs):
    # if os.path.exists("embeddings/image_emb.npy"):
    #     print("Loading cached image embeddings...")
    #     image_embeddings = np.load("embeddings/image_emb.npy", allow_pickle=True)
    #     image_embeddings = np.squeeze(image_embeddings)
    #     assert image_embeddings.ndim == 2, f"Expected 2D, got {image_embeddings.shape}"
    #     return image_embeddings.tolist()

    print("Computing image embeddings...")
    image_embeddings = embedder.embed_images(image_docs)

    # image_embeddings = np.array(image_embeddings)
    # image_embeddings = np.squeeze(image_embeddings)        # fix before saving

    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/image_emb.npy", image_embeddings)

    return image_embeddings.tolist()


def build_db(db_type, index_type):
    if db_type == "chroma":
        db = ChromaDB()
        # db.collection.delete()
        return db
    elif db_type == "qdrant":
        return QdrantDB(dim=EMBED_DIM)
    elif db_type == "milvus":
        # db.drop()
        return MilvusDB(dim=EMBED_DIM, index_type=index_type)
    else:
        raise ValueError(f"Unknown db_type: {db_type}")


def index_data(db, text_ids, text_embeddings, text_chunks,
               image_ids, image_embeddings, image_texts):

    ## ---- For text only ---
    # all_embeddings = text_embeddings + image_embeddings
    # all_embeddings = np.vstack([text_embeddings, image_embeddings])
    # all_ids = text_ids + image_ids
    # all_payloads = text_chunks + image_docs
    # db.add(all_ids, all_embeddings, text_chunks)

    db.add(text_ids, text_embeddings, text_chunks)
    db.add(image_ids, image_embeddings, image_texts)


def main(db_type="chroma", index_type="HNSW"):

    conn = init_db()

    text_chunks = load_text_chunks()
    image_docs  = load_image_docs()

    embedder = build_embedder()

    text_embeddings  = get_text_embeddings(embedder, text_chunks)
    image_embeddings = get_image_embeddings(embedder, image_docs)

    text_ids    = [f"text_{i}" for i in range(len(text_chunks))]
    # text_ids = [str(i) for i in range(len(text_chunks))]
    image_ids   = [doc["id"] for doc in image_docs]
    image_texts = [doc["content"] for doc in image_docs]


    db = build_db(db_type, index_type)
    index_data(db, text_ids, text_embeddings, text_chunks,
               image_ids, image_embeddings, image_texts)

    queries = load_queries(QUERY_PATH)
    results = run_queries(db, embedder, queries, db_type)

    display_summary(results)
    log_result(conn, db_type, index_type, results)
    print(conn.execute("SELECT * FROM results").fetchdf())


if __name__ == "__main__":

    # for db in ["chroma", "qdrant", "milvus"]:
    # for db in ["chroma", "qdrant"]:
    #     print(f"Running benchmark for: {db}")
    #     main(db, index_type="HNSW")

    db = "milvus"
    print(f"Running benchmark for: {db}")
    # main(db, index_type="DISKANN")
    # main(db, index_type="HNSW")
    main(db, index_type="IVF_FLAT")