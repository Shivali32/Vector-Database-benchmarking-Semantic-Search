import time
import numpy as np


def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def extract_ids(response, db_type):
    if db_type == "chroma":
        return response["ids"][0]
    elif db_type == "qdrant":
        return [str(hit.id) for hit in response]
    elif db_type == "milvus":
        return [str(hit.id) for hit in response]
    

def extract_texts(response, db_type):
    if db_type == "chroma":
        return response["documents"][0]
    elif db_type == "qdrant":
        return [
            hit.payload.get("text", str(hit.payload))
            for hit in response
        ]
    elif db_type == "milvus":
        return [
            hit.entity.get("text") or str(hit.entity)
            for hit in response
        ]


def extract_vectors(response, db_type):
    if db_type == "chroma":
        # chroma returns embeddings as list of lists under "embeddings"
        return response["embeddings"][0]
    elif db_type == "qdrant":
        # qdrant returns hit.vector directly since with_vectors=True
        return [hit.vector for hit in response]
    elif db_type == "milvus":
         return [hit["vector"] for hit in response]


def compute_recall(query_emb, retrieved_vectors, threshold=0.55):
    for vec in retrieved_vectors:
        if vec is None:
            continue
        score = cosine_similarity(query_emb, vec)
        if score >= threshold:
            return 1
    return 0


def run_queries(db, embedder, queries, db_type, k=3):
    total_start = time.time()

    query_texts  = [item["query"]  for item in queries]
    answer_texts = [item["answer"] for item in queries]

    query_embeddings  = embedder.embed_queries(
        query_texts,
        save_path="embeddings/query_embeddings.npy"
    )

    answer_embeddings = embedder.embed_queries(
        answer_texts,
        save_path="embeddings/answer_embeddings.npy"
    )

    total_recall = 0
    latencies = []

    for q_emb, gt_emb in zip(query_embeddings, answer_embeddings):
        start = time.time()

        if db_type == "chroma":
            response = db.collection.query(
                query_embeddings=[q_emb],
                n_results=k,
                include=["documents", "embeddings"]
            )
        else:
            response = db.query(q_emb, k)

        # latency = time.time() - start
        # total_latency += latency
        latencies.append(time.time() - start)

        retrieved_vectors = extract_vectors(response, db_type)
        recall = compute_recall(gt_emb, retrieved_vectors)
        total_recall += recall

    total_time = time.time() - total_start
    total_queries = len(queries)
    latencies_array = np.array(latencies)

    metrics = {
        "queries": total_queries,
        "total_time": round(total_time, 2),
        "avg_latency": round(latencies_array.mean(), 4),
        "p50_latency": round(np.percentile(latencies_array, 50), 4),
        "p95_latency": round(np.percentile(latencies_array, 95), 4),
        "p99_latency": round(np.percentile(latencies_array, 99), 4),
        "throughput": round(total_queries / total_time, 2),
        "recall_k": round(total_recall / total_queries, 4)
    }
    return metrics
