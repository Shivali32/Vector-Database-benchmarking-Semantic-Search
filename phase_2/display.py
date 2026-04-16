def display_summary(metrics):
    print(f"Total queries        : {metrics['queries']}")
    print(f"Total time (s)       : {metrics['total_time']}")

    print(f"\n-- Indexing --")
    print(f"Text embed time (s)  : {metrics.get('text_embed_time', 'N/A')}")
    print(f"Image embed time (s) : {metrics.get('image_embed_time', 'N/A')}")
    print(f"Index insert time (s): {metrics.get('index_time', 'N/A')}")

    print(f"\n-- Query --")
    print(f"Query embed time (s) : {metrics.get('query_embed_time', 'N/A')}")
    print(f"Answer embed time (s): {metrics.get('answer_embed_time', 'N/A')}")
    print(f"Avg retrieval (s)    : {metrics['avg_latency']}")
    print(f"Avg scoring (s)      : {metrics.get('avg_scoring_time', 'N/A')}")
    print(f"p50 / p95 / p99 (s)  : {metrics['p50_latency']} / {metrics['p95_latency']} / {metrics['p99_latency']}")

    print(f"\n-- Results --")
    print(f"Throughput (q/s)     : {metrics['throughput']}")
    print(f"Recall@K             : {metrics['recall_k']}")
    print(f"Precision@K          : {metrics.get('precision_k', 'N/A')}")



# def display_summary(metrics):
    # total_latency = sum(r["latency"] for r in results)
    # print(f"Total Queries: {len(results)}")
    # print(f"Average Latency: {round(total_latency/len(results),4)} sec")
    # print(f"Recall@k: {round(avg_recall,4)}")
    
    # print(f"Total queries        : {metrics['queries']}")
    # print(f"Total time (s)       : {metrics['total_time']}")
    # print(f"Avg latency (s)      : {metrics['avg_latency']}")
    # print(f"Throughput (q/s)     : {metrics['throughput']}")
    # print(f"Recall@k             : {metrics['recall_k']}")
    # print(f"Precision@k          : {metrics['precision_k']}")
