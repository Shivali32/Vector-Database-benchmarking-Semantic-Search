import time

def extract_texts(response, db_type):
    if db_type == "chroma":
        return response["documents"][0]
    elif db_type == "qdrant":
        return [hit.payload["text"] for hit in response]
    elif db_type == "milvus":
        return [hit.entity.get("text") for hit in response]

def compute_recall(query, retrieved_docs):    
    relevant = 0
    for doc in retrieved_docs:
        if query.lower() in doc.lower():
            relevant = 1
            break
    return relevant

def run_queries(db, embedder, queries, db_type, k=3):
    total_start = time.time()

    query_embeddings = embedder.embed_documents(queries)
    total_recall = 0
    total_latency = 0

    for q, emb in zip(queries, query_embeddings):
        start = time.time()
        response = db.query(emb, k)
        latency = time.time() - start
        total_latency += latency

        docs = extract_texts(response, db_type)
        recall = compute_recall(q, docs)
        total_recall += recall

    total_time = time.time() - total_start
    total_queries = len(queries)

    metrics = {
        "queries": total_queries,
        "total_time": round(total_time, 2),
        "avg_latency": round(total_latency / total_queries, 4),
        "throughput": round(total_queries / total_time, 2),
        "recall_k": round(total_recall / total_queries, 4)
    }

    return metrics
