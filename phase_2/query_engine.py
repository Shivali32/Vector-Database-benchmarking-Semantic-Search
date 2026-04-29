import time
import numpy as np

# from phase_2 import embedder


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

# def compute_recall_at_k(query_emb, retrieved_vectors, k=3):
#     """
#     Recall@K: Average cosine similarity of top-K results to query.
    
#     Higher is better. Range: [0, 1]
#     This measures: "How similar are my top-K results to the query?"
    
#     Args:
#         query_emb: Query embedding vector
#         retrieved_vectors: List of retrieved document embeddings
#         k: Number of top results to evaluate
    
#     Returns:
#         Average similarity score
#     """
#     if not retrieved_vectors:
#         return 0.0
    
#     similarities = []
#     for vec in retrieved_vectors[:k]:
#         if vec is not None:
#             sim = cosine_similarity(query_emb, vec)
#             similarities.append(sim)
    
#     return sum(similarities) / len(similarities) if similarities else 0.0


# def compute_precision_at_k(query_emb, retrieved_vectors, threshold=0.5, k=3):
#     """
#     Precision@K: Percentage of top-K results above similarity threshold.
    
#     Higher is better. Range: [0, 1]
#     This measures: "What fraction of my top-K results are highly relevant?"
    
#     Args:
#         query_emb: Query embedding vector
#         retrieved_vectors: List of retrieved document embeddings
#         threshold: Minimum similarity to be considered "relevant"
#         k: Number of top results to evaluate
    
#     Returns:
#         Fraction of results above threshold
#     """
#     if not retrieved_vectors:
#         return 0.0
    
#     relevant_count = 0
#     total_count = 0
    
#     for vec in retrieved_vectors[:k]:
#         if vec is not None:
#             total_count += 1
#             sim = cosine_similarity(query_emb, vec)
#             if sim >= threshold:
#                 relevant_count += 1
    
#     return relevant_count / total_count if total_count > 0 else 0.0

def compute_recall_at_k(answer_emb, retrieved_vectors, k=3):
    """
    Check if the ANSWER embedding is similar to any retrieved chunk.
    """
    max_similarity = 0
    threshold = 0.6
    
    for vec in retrieved_vectors[:k]:
        if vec is not None:
            # Compare ANSWER vs RETRIEVED (not query vs retrieved)
            sim = cosine_similarity(answer_emb, vec)

    #         if sim >= threshold:
    #             return 1
    # return 0
            max_similarity = max(max_similarity, sim)            
    return max_similarity  # Higher = better


def compute_precision_at_k(answer_emb, retrieved_vectors, threshold=0.7, k=3):
    """
    How many retrieved chunks match the answer?
    """
    relevant = 0
    
    for vec in retrieved_vectors[:k]:
        if vec is not None:
            sim = cosine_similarity(answer_emb, vec)
            if sim >= threshold:
                relevant += 1
    
    return relevant / k


# def compute_recall(query_emb, retrieved_vectors, threshold=0.6):
#     # sims = [cosine_similarity(query_emb, v) for v in retrieved_vectors if v is not None]
#     # return max(sims) if sims else 0
#     for vec in retrieved_vectors:
#         if vec is None:
#             continue
#         score = cosine_similarity(query_emb, vec)
#         if score >= threshold:
#             return 1
#     return 0

# def compute_precision(query_emb, retrieved_vectors, threshold=0.6):
#     relevant = 0
#     total = 0

#     for vec in retrieved_vectors:
#         if vec is None:
#             continue

#         total += 1
#         score = cosine_similarity(query_emb, vec)

#         if score >= threshold:
#             relevant += 1

#     return relevant / total if total > 0 else 0

def run_queries(db, embedder, queries, db_type, k=3):
    total_start = time.time()

    query_texts  = [item["query"]  for item in queries]

    # ---- Query embedding time ----
    t0 = time.time()
    query_embeddings = embedder.embed_queries(
        query_texts,
        save_path=None,
        use_cache=False
    )
    query_embed_time = round(time.time() - t0, 4)

    total_recall = 0
    total_precision = 0
    retrieval_latencies = []
    scoring_latencies   = []

    t0 = time.time()
    answer_texts = [item["answer"] for item in queries]
    answer_embeddings = embedder.embed_queries(answer_texts, save_path=None, use_cache=False)
    answer_embed_time = round(time.time() - t0, 4)

    # for idx, q_emb in enumerate(query_embeddings):
    for idx, (q_emb, ans_emb) in enumerate(zip(query_embeddings, answer_embeddings)):

        # ---- Retrieval latency (DB query) ----
        t0 = time.time()
        if db_type == "chroma":
            response = db.text_collection.query(
                query_embeddings=[q_emb],
                n_results=k,
                include=["embeddings"]  # Need embeddings for similarity
            )
        else:
            response = db.query(q_emb, k)
        retrieval_latencies.append(time.time() - t0)

        # ---- Scoring time (metrics compute) ----
        t0 = time.time()
        retrieved_vectors = extract_vectors(response, db_type)
        
        recall = compute_recall_at_k(ans_emb, retrieved_vectors, k=k)
        precision = compute_precision_at_k(ans_emb, retrieved_vectors, threshold=0.4, k=k)
        # recall = compute_recall_at_k(q_emb, retrieved_vectors, k=k)
        # precision = compute_precision_at_k(q_emb, retrieved_vectors, threshold=0.5, k=k)
        
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
        "avg_embed_time":     query_embed_time,  # Only query embeddings now

        # retrieval latency (per query)
        "avg_latency":        round(ret_arr.mean(), 4),
        "p50_latency":        round(np.percentile(ret_arr, 50), 4),
        "p95_latency":        round(np.percentile(ret_arr, 95), 4),
        "p99_latency":        round(np.percentile(ret_arr, 99), 4),

        # scoring latency (per query)
        "avg_scoring_time":   round(score_arr.mean(), 6),

        "throughput":         round(total_queries / total_time, 2),
        "recall_k":           round(total_recall / total_queries, 4),
        "precision_k":        round(total_precision / total_queries, 4)
    }
    return metrics

# def run_queries(db, embedder, queries, db_type, k=3):
#     total_start = time.time()

#     query_texts  = [item["query"]  for item in queries]
#     answer_texts = [item["answer"] for item in queries]

#     # ---- Step 1: Query embedding time ----
#     t0 = time.time()
#     query_embeddings = embedder.embed_queries(
#         query_texts,
#         save_path="embeddings/query_embeddings.npy"
#     )
#     query_embed_time = round(time.time() - t0, 4)

#     # ---- Step 2: Answer embedding time ----
#     t0 = time.time()
#     answer_embeddings = embedder.embed_queries(
#         answer_texts,
#         save_path="embeddings/answer_embeddings.npy"
#     )
#     answer_embed_time = round(time.time() - t0, 4)    

#     total_recall = 0
#     total_precision = 0
#     retrieval_latencies = []
#     scoring_latencies   = []

#     for q_emb, gt_emb in zip(query_embeddings, answer_embeddings):

#         # ---- Step 3: Retrieval latency (DB query) ----
#         t0 = time.time()
#         if db_type == "chroma":
#             response = db.text_collection.query(
#                 query_embeddings=[q_emb],
#                 n_results=k,
#                 include=["documents", "embeddings"]
#             )
#         else:
#             response = db.query(q_emb, k)
#         retrieval_latencies.append(time.time() - t0)

#         # ---- Step 4: Scoring time (recall + precision compute) ----
#         t0 = time.time()
#         retrieved_vectors = extract_vectors(response, db_type)
#         recall    = compute_recall(gt_emb, retrieved_vectors)
#         precision = compute_precision(gt_emb, retrieved_vectors)
#         scoring_latencies.append(time.time() - t0)

#         total_recall    += recall
#         total_precision += precision

#     total_time      = time.time() - total_start
#     total_queries   = len(queries)
#     ret_arr         = np.array(retrieval_latencies)
#     score_arr       = np.array(scoring_latencies)

#     metrics = {
#         "queries":            total_queries,
#         "total_time":         round(total_time, 4),

#         # embedding timings
#         "query_embed_time":   query_embed_time,
#         "answer_embed_time":  answer_embed_time,
#         "avg_embed_time":   round((query_embed_time + answer_embed_time) / 2, 4),

#         # retrieval latency (per query)
#         "avg_latency":        round(ret_arr.mean(), 4),
#         "p50_latency":        round(np.percentile(ret_arr, 50), 4),
#         "p95_latency":        round(np.percentile(ret_arr, 95), 4),
#         "p99_latency":        round(np.percentile(ret_arr, 99), 4),

#         # scoring latency (per query)
#         "avg_scoring_time":   round(score_arr.mean(), 6),

#         "throughput":         round(total_queries / total_time, 2),
#         "recall_k":           round(total_recall    / total_queries, 4),
#         "precision_k":        round(total_precision / total_queries, 4)
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

