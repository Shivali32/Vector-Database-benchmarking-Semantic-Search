import duckdb

def init_db(db_path="benchmark.duckdb"):
    conn = duckdb.connect(db_path)
    # conn = duckdb.connect("benchmark.duckdb")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS results (
        run_id TIMESTAMP,
        db_name VARCHAR,
        index_type VARCHAR,
        queries INTEGER,
        total_time DOUBLE,
        avg_latency DOUBLE,
        throughput DOUBLE,
        recall_k DOUBLE,
        precision_k DOUBLE,
        p50_latency DOUBLE,
        p95_latency DOUBLE,
        p99_latency DOUBLE
    )
    """)

    return conn


def log_result(conn, db_name, index_type, metrics):
    conn.execute("""
        INSERT INTO results VALUES (
            CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
    """, (
        db_name, index_type,
        metrics["queries"],
        metrics["total_time"],
        metrics["avg_latency"],
        metrics["throughput"],
        metrics["recall_k"],
        metrics["p50_latency"],
        metrics["p95_latency"],
        metrics["p99_latency"],
        metrics.get("query_embed_time"),
        metrics.get("answer_embed_time"),
        metrics.get("avg_scoring_time"),
        metrics.get("text_embed_time"),
        metrics.get("image_embed_time"),
        metrics.get("index_time"),
    ))

    