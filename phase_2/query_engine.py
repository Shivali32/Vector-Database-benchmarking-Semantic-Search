import time
import numpy as np


def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def extract_ids(response, db_type):
    if db_type == "chroma":
        return response["ids"][0]
    elif db_type == "qdrant":
        return [hit.payload.get("original_id") for hit in response]
        # return [str(hit.id) for hit in response]
    elif db_type == "milvus":
        return [hit["original_id"] for hit in response]
        # return [str(hit.id) for hit in response]
    

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


def compute_recall(query_emb, retrieved_vectors, threshold=0.6):
    # sims = [cosine_similarity(query_emb, v) for v in retrieved_vectors if v is not None]
    # return max(sims) if sims else 0
    for vec in retrieved_vectors:
        if vec is None:
            continue
        score = cosine_similarity(query_emb, vec)
        if score >= threshold:
            return 1
    return 0

def compute_precision(query_emb, retrieved_vectors, threshold=0.6):
    relevant = 0
    total = 0

    for vec in retrieved_vectors:
        if vec is None:
            continue

        total += 1
        score = cosine_similarity(query_emb, vec)

        if score >= threshold:
            relevant += 1

    return relevant / total if total > 0 else 0

def run_queries(db, embedder, queries, db_type, k=3):
    total_start = time.time()

    query_texts  = [item["query"]  for item in queries]
    answer_texts = [item["answer"] for item in queries]

    # ---- Step 1: Query embedding time ----
    t0 = time.time()
    query_embeddings = embedder.embed_queries(
        query_texts,
        save_path="embeddings/query_embeddings.npy"
    )
    query_embed_time = round(time.time() - t0, 4)

    # ---- Step 2: Answer embedding time ----
    t0 = time.time()
    answer_embeddings = embedder.embed_queries(
        answer_texts,
        save_path="embeddings/answer_embeddings.npy"
    )
    answer_embed_time = round(time.time() - t0, 4)

    total_recall = 0
    total_precision = 0
    retrieval_latencies = []
    scoring_latencies   = []

    for q_emb, gt_emb in zip(query_embeddings, answer_embeddings):

        # ---- Step 3: Retrieval latency (DB query) ----
        t0 = time.time()
        if db_type == "chroma":
            response = db.text_collection.query(
                query_embeddings=[q_emb],
                n_results=k,
                include=["documents", "embeddings"]
            )
        else:
            response = db.query(q_emb, k)
        retrieval_latencies.append(time.time() - t0)

        # ---- Step 4: Scoring time (recall + precision compute) ----
        t0 = time.time()
        retrieved_vectors = extract_vectors(response, db_type)
        recall    = compute_recall(gt_emb, retrieved_vectors)
        precision = compute_precision(gt_emb, retrieved_vectors)
        scoring_latencies.append(time.time() - t0)

        total_recall    += recall
        total_precision += precision

    total_time      = time.time() - total_start
    total_queries   = len(queries)
    ret_arr         = np.array(retrieval_latencies)
    score_arr       = np.array(scoring_latencies)

    metrics = {
        "queries":            total_queries,
        "total_time":         round(total_time, 4),

        # embedding timings
        "query_embed_time":   query_embed_time,
        "answer_embed_time":  answer_embed_time,
        "avg_embed_time":   round((query_embed_time + answer_embed_time) / 2, 4),

        # retrieval latency (per query)
        "avg_latency":        round(ret_arr.mean(), 4),
        "p50_latency":        round(np.percentile(ret_arr, 50), 4),
        "p95_latency":        round(np.percentile(ret_arr, 95), 4),
        "p99_latency":        round(np.percentile(ret_arr, 99), 4),

        # scoring latency (per query)
        "avg_scoring_time":   round(score_arr.mean(), 6),

        "throughput":         round(total_queries / total_time, 2),
        "recall_k":           round(total_recall    / total_queries, 4),
        "precision_k":        round(total_precision / total_queries, 4)
    }
    return metrics

# def run_queries(db, embedder, queries, db_type, k=3):
#     total_start = time.time()

#     query_texts  = [item["query"]  for item in queries]
#     answer_texts = [item["answer"] for item in queries]

#     query_embeddings  = embedder.embed_queries(
#         query_texts,
#         save_path="embeddings/query_embeddings.npy"
#     )

#     answer_embeddings = embedder.embed_queries(
#         answer_texts,
#         save_path="embeddings/answer_embeddings.npy"
#     )

#     total_recall = 0
#     total_precision = 0
#     latencies = []

#     for q_emb, gt_emb in zip(query_embeddings, answer_embeddings):
#         start = time.time()

#         if db_type == "chroma":
#             response = db.text_collection.query(
#                 query_embeddings=[q_emb],
#                 n_results=k,
#                 include=["documents", "embeddings"]
#             )
#             # response = db.collection.query(
#             #     query_embeddings=[q_emb],
#             #     n_results=k,
#             #     include=["documents", "embeddings"]
#             # )
#         else:
#             response = db.query(q_emb, k)

#         # latency = time.time() - start
#         # total_latency += latency
#         latencies.append(time.time() - start)

#         retrieved_vectors = extract_vectors(response, db_type)
        
#         recall = compute_recall(gt_emb, retrieved_vectors)
#         total_recall += recall
        
#         precision = compute_precision(gt_emb, retrieved_vectors)
#         total_precision += precision

#     total_time = time.time() - total_start
#     total_queries = len(queries)
#     latencies_array = np.array(latencies)

#     metrics = {
#         "queries": total_queries,
#         "total_time": round(total_time, 2),
#         "avg_latency": round(latencies_array.mean(), 4),
#         "p50_latency": round(np.percentile(latencies_array, 50), 4),
#         "p95_latency": round(np.percentile(latencies_array, 95), 4),
#         "p99_latency": round(np.percentile(latencies_array, 99), 4),
#         "throughput": round(total_queries / total_time, 2),
#         "recall_k": round(total_recall / total_queries, 4),
#         "precision_k": round(total_precision / total_queries, 4)
#     }
#     return metrics


def rerank_chunks(query_emb, chunk_ids, chunk_texts, embedder, top_k=3):
    chunk_embs = embedder.embed_documents(chunk_texts, save_path=None)

    scores = []

    for cid, emb, text in zip(chunk_ids, chunk_embs, chunk_texts):
        score = cosine_similarity(query_emb, emb)
            
        # if "was" in text[:100] and len(text) < 600:
        #     score += 0.1
        
        scores.append((score, cid, text))

    scores.sort(reverse=True, key=lambda x: x[0])
    top = scores[:top_k]

    return [t[2] for t in top]

