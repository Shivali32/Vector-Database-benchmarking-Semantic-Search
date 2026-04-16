from data_loader import load_documents, load_wit_images, chunk_text, os
from chunk_db import init_chunk_db, insert_text_chunks, insert_image_chunks
from chunk_db import init_chunk_db, load_text_chunks_from_db, load_image_chunks_from_db, fetch_chunks_by_ids
from embedder import Embedder, np
from model_loader import load_model
from vector_db.chroma_db import ChromaDB
from vector_db.qdrant_db import QdrantDB
from vector_db.milvus_db import MilvusDB
from query_loader import load_queries
from query_engine import extract_texts, run_queries, extract_ids, rerank_chunks, cosine_similarity
from display import display_summary
from db_logger import init_db, log_result
from rag import RAGPipeline

import random, time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from PIL import Image
import matplotlib.pyplot as plt


TEXT_DATA_PATH   = "wiki_dataset"
IMAGE_META_PATH  = "wit_metadata/wit_subset_metadata.json"
QUERY_PATH       = "queries/queries_marco.json"
EMBED_DIM        = 384      # 512
conn = init_db()

# text_data_path = "wiki_sub"
# image_metadata_path = "wit_metadata/wit_meta_sub.json"


def load_text_chunks():
    text_docs = load_documents(TEXT_DATA_PATH)

    text_chunks = []
    chunk_counter = 0

    for doc in text_docs:
        chunks = chunk_text(doc["content"])

        for i, chunk in enumerate(chunks):
            text_chunks.append({
                "id": f"text_{chunk_counter}",
                "doc_id": doc["doc_id"],
                "content": chunk,
                "is_intro": (i == 0)
            })
            chunk_counter += 1

    # text_chunks = text_chunks[:2]
    return text_chunks


def load_image_docs():
    image_docs = load_wit_images(IMAGE_META_PATH)
    # image_docs = image_docs[:2]
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
        # client = db.collection._client
        # client.delete_collection(db.collection.name)
        # db.collection = client.create_collection(db.collection.name)        
        return db        
    elif db_type == "qdrant":
        return QdrantDB(dim=EMBED_DIM)
    elif db_type == "milvus":
        # db.drop()
        return MilvusDB(dim=EMBED_DIM, index_type=index_type)
    else:
        raise ValueError(f"Unknown db_type: {db_type}")


def index_data(db, text_ids, text_embeddings, text_chunks,
               image_ids, image_embeddings, image_texts, image_metadatas):

    db.add_text(text_ids, text_embeddings, text_chunks)
    db.add_image(image_ids, image_embeddings, image_texts, image_metadatas)

def clear_chunks_table(conn):
    conn.execute("DELETE FROM chunks")
    conn.commit()

def get_chunks(store_chunks):
    
    text_chunks = None
    image_docs = None

    conn_chunks = init_chunk_db()
    if store_chunks:
        text_chunks = load_text_chunks()
        image_docs = load_image_docs()

        clear_chunks_table(conn_chunks)

        insert_text_chunks(conn_chunks, text_chunks)
        insert_image_chunks(conn_chunks, image_docs)

    else:
        text_chunks = load_text_chunks_from_db(conn_chunks)
        image_docs = load_image_chunks_from_db(conn_chunks)

        # image_docs = [
        #     {"id": id_, "content": text}
        #     for id_, text in zip(image_ids, image_texts)
        # ]

    conn_chunks.close()    

    return text_chunks,image_docs

def rag_top_k(sample_queries, db, embedder, db_type, k=3):
    rag = RAGPipeline()

    conn_chunks = init_chunk_db()
    results_data = []
    print("\n------- Sample Queries -------\n")

    for item in sample_queries:
        query = item["query"]
        query_embedding = embedder.embed_queries([query],
                          save_path="embeddings/query_embeddings.npy",
                          use_cache=False)[0]

        if db_type == "chroma":
            
            response = db.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
            )
            # response = db.collection.query(
            #     query_embeddings=[query_embedding],
            #     n_results=k,
            #     # include=["ids"]
            # )
            # print(response)
        else:
            response = db.query(query_embedding, k, modality="text")
            
        # retrieved_chunks = extract_texts(response, db_type)
        top_ids = extract_ids(response, db_type)
        
        retrieved_chunks = fetch_chunks_by_ids(conn_chunks, top_ids)
        retrieved_chunks = fetch_chunks_by_ids(conn_chunks, top_ids)

        retrieved_chunks = rerank_chunks( query_embedding, top_ids, retrieved_chunks, embedder, top_k=3)

        answer = rag.generate_answer(query, retrieved_chunks)

        print("\nQuery:", query)        
        
        # print("\nRetrieved Chunks:")
        # for chunk in retrieved_chunks:
        #     print("----", chunk[:200])

        print("\n====================Generated Answer:=====================\n")
        print(answer)
        # print("\n" + "="*50)


        results_data.append({
            "query": query,
            "chunks": retrieved_chunks,
            "answer": answer
        })
    
    conn_chunks.close()
    return results_data

def run_image_queries(db, embedder, queries, k=3):
    print("\n------ IMAGE RETRIEVAL ------\n")

    for q in queries:
        query = q["query"]

        q_emb = embedder.embed_image_query(query)

        response = db.image_collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        metas = response["metadatas"][0]
        captions = response["documents"][0]
        
        scores = []

        for meta, caption in zip(metas, captions):
            clean_caption = caption[:200] 
            cap_emb = embedder.embed_image_query(clean_caption)
            score = cosine_similarity(q_emb, cap_emb)
            scores.append((score, meta))

        scores.sort(reverse=True, key=lambda x: x[0])

        best_meta = scores[0][1]

        print(f"\nQuery: {query}")
        print("Best Image:", best_meta["image_id"])
        print("Caption   :", best_meta["caption"])


def main(db_type="chroma", index_type="HNSW", store_chunks=True):

    text_chunks, image_docs = get_chunks(store_chunks)
    # print(f"Loaded text chunks {text_chunks}")
    # print(f"Loaded image chunks {image_docs}")

    embedder = build_embedder()

    # ---- text embedding time ----
    t0 = time.time()
    text_embeddings = get_text_embeddings(embedder, text_chunks)
    text_embed_time = round(time.time() - t0, 4)

    # ---- image embedding time ----
    t0 = time.time()
    image_embeddings = get_image_embeddings(embedder, image_docs)
    image_embed_time = round(time.time() - t0, 4)

    # text_ids    = [f"text_{i}" for i in range(len(text_chunks))]
    # text_ids = [str(i) for i in range(len(text_chunks))]

    text_ids = [doc["id"] for doc in text_chunks]
    text_chunks = [doc["content"] for doc in text_chunks]

    image_ids   = [doc["id"] for doc in image_docs]
    image_texts = [doc["content"] for doc in image_docs]
    
    image_metadatas = [
        {
            "image_id": doc["id"] or "",
            "image_path": doc["image_path"] or "",
            "caption": doc["content"] or ""
        }
        for doc in image_docs
    ]   

    db = build_db(db_type, index_type)

    # ---- index time ----
    t0 = time.time()
    index_data(db, text_ids, text_embeddings, text_chunks,
               image_ids, image_embeddings, image_metadatas, image_texts)
    index_time = round(time.time() - t0, 4)

    queries = load_queries(QUERY_PATH)
    results = run_queries(db, embedder, queries, db_type, k=3)
        
    # attach indexing timings to the same metrics dict
    results["text_embed_time"]  = text_embed_time
    results["image_embed_time"] = image_embed_time
    results["index_time"]       = index_time

    # sample_queries = random.sample(queries, 3)
    sample_queries = [
        { 'query': "who are mohicans", 'answer': '' },
        { 'query': 'where is pembroke', 'answer': '' },
        { 'query': 'who is Jackson Beardy', 'answer': ''},
    ]
    # print(sample_queries)
    rag_top_k(sample_queries, db, embedder, db_type, k=3)

    image_queries = [
        {"query": "coral reef wakatobi"},
        {"query": "volcano crater aerial view"},
        {"query": "malaysian lamb curry"}
    ]

    run_image_queries(db, embedder, image_queries, k=3)

    display_summary(results)
    log_result(conn, db_type, index_type, results)


if __name__ == "__main__":

    # for db in ["chroma", "qdrant", "milvus"]:
    for db in [
        "chroma", 
        # "qdrant"
    ]:
        print(f"Running benchmark for: {db}")
        main(db, index_type="HNSW")

    # db = "milvus"
    # print(f"Running benchmark for: {db}")
    # main(db, index_type="DISKANN")
    # main(db, index_type="HNSW")
    # main(db, index_type="IVF_FLAT")

    # print(conn.execute("SELECT * FROM results ORDER BY run_id DESC limit 10").fetchdf())