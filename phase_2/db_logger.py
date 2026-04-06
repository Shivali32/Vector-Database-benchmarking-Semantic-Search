import duckdb

def init_db(db_path="benchmark.duckdb"):
    conn = duckdb.connect(db_path)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS results (
        run_id TIMESTAMP,
        db_name VARCHAR,
        index_type VARCHAR,
        queries INTEGER,
        total_time DOUBLE,
        avg_latency DOUBLE,
        throughput DOUBLE,
        recall_k DOUBLE
    )
    """)

    return conn

def log_result(conn, db_name, index_type, metrics):
    conn.execute("""
        INSERT INTO results VALUES (
            CURRENT_TIMESTAMP,
            ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        db_name,
        index_type,
        metrics["queries"],
        metrics["total_time"],
        metrics["avg_latency"],
        metrics["throughput"],
        metrics["recall_k"]
    ))

    