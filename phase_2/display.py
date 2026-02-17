def display_summary(metrics):
    # total_latency = sum(r["latency"] for r in results)
    # print(f"Total Queries: {len(results)}")
    # print(f"Average Latency: {round(total_latency/len(results),4)} sec")
    # print(f"Recall@k: {round(avg_recall,4)}")
    
    print(f"Total queries        : {metrics['Total queries']}")
    print(f"Total time (s)       : {metrics['Total time (s)']}")
    print(f"Avg latency (s)      : {metrics['Avg latency (s)']}")
    print(f"Throughput (q/s)     : {metrics['Throughput (q/s)']}")
    print(f"Recall@k             : {metrics['Recall@k']}")
