from data_loader import load_documents, load_wit_images, chunk_text, os
from chunk_db import init_chunk_db, insert_text_chunks, insert_image_chunks
from chunk_db import init_chunk_db, load_text_chunks_from_db, load_image_chunks_from_db
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
conn = init_db()

# text_data_path = "wiki_sub"
# image_metadata_path = "wit_metadata/wit_meta_sub.json"


def load_text_chunks():
    text_docs = load_documents(TEXT_DATA_PATH)

    text_chunks = []
    chunk_counter = 0

    for doc in text_docs:
        chunks = chunk_text(doc["content"])

        for chunk in chunks:
            text_chunks.append({
                "id": f"text_{chunk_counter}", 
                "doc_id": doc["doc_id"],
                "content": chunk
            })
            chunk_counter += 1

    return text_chunks


def load_image_docs():
    image_docs = load_wit_images(IMAGE_META_PATH)
    image_docs = image_docs[:5]
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

    # print("Computing text embeddings...")
    text_embeddings = embedder.embed_documents(text_chunks)

    return text_embeddings.tolist()


def get_image_embeddings(embedder, image_docs):

    # print("Computing image embeddings...")
    image_embeddings = embedder.embed_images(image_docs)

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

def get_chunks(store_chunks):
    
    text_chunks = None
    image_docs = None

    conn_chunks = init_chunk_db()
    if store_chunks:
        text_chunks = load_text_chunks()
        image_docs = load_image_docs()

        insert_text_chunks(conn_chunks, text_chunks)
        insert_image_chunks(conn_chunks, image_docs)

    else:
        text_chunks = load_text_chunks_from_db(conn_chunks)
        image_ids, image_texts = load_image_chunks_from_db(conn_chunks)

        image_docs = [
            {"id": id_, "content": text}
            for id_, text in zip(image_ids, image_texts)
        ]

    conn_chunks.close()    

    return text_chunks, image_docs


def main(db_type="chroma", index_type="HNSW", store_chunks=False):

    text_chunks, image_docs = get_chunks(store_chunks)
    print(f"Loaded text chunks {text_chunks}")
    print(f"Loaded image chunks {image_docs}")

    embedder = build_embedder()

    text_embeddings  = get_text_embeddings(embedder, text_chunks)
    image_embeddings = get_image_embeddings(embedder, image_docs)

    # text_ids    = [f"text_{i}" for i in range(len(text_chunks))]
    # text_ids = [str(i) for i in range(len(text_chunks))]

    text_ids = [doc["id"] for doc in text_chunks]
    text_chunks = [doc["content"] for doc in text_chunks]
    text_doc_ids = [doc["doc_id"] for doc in text_chunks]

    image_ids   = [doc["id"] for doc in image_docs]
    image_texts = [doc["content"] for doc in image_docs]


    db = build_db(db_type, index_type)
    index_data(db, text_ids, text_embeddings, text_chunks,
                image_ids, image_embeddings, image_texts)

    queries = load_queries(QUERY_PATH)
    results = run_queries(db, embedder, queries, db_type, k=5)

    display_summary(results)
    log_result(conn, db_type, index_type, results)


if __name__ == "__main__":

    # for db in ["chroma", "qdrant", "milvus"]:
    for db in ["chroma"]:
        print(f"Running benchmark for: {db}")
        main(db, index_type="HNSW", store_chunks=True)

    # db = "milvus"
    # print(f"Running benchmark for: {db}")
    # main(db, index_type="DISKANN")
    # main(db, index_type="HNSW")
    # main(db, index_type="IVF_FLAT")

    # print(conn.execute("SELECT * FROM results ORDER BY run_id DESC limit 10").fetchdf())