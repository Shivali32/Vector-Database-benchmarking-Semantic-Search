import duckdb

def init_chunk_db():
    conn = duckdb.connect("chunks.duckdb")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            type TEXT,
            content TEXT,
            image_path TEXT
        )
    """)

    return conn

def insert_text_chunks(conn, text_chunks):
    data = []

    for i, chunk in enumerate(text_chunks):
        data.append((
            f"text_{i}",    
            f"doc_{i}",     
            "text",
            chunk,
            None
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)
    """, data)

    print(f"Inserted {len(data)} text chunks")


def insert_image_chunks(conn, image_docs):
    data = []

    for doc in image_docs:
        data.append((
            doc["id"],
            doc["id"],
            "image",
            doc["content"],
            doc["image_path"]
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)
    """, data)

    print(f"Inserted {len(data)} image chunks")

def load_text_chunks_from_db(conn):
    result = conn.execute("""
        SELECT content FROM chunks WHERE type='text' ORDER BY chunk_id
    """).fetchall()

    print(f"Loaded {len(result)} text chunks")

    return [row[0] for row in result]


def load_image_chunks_from_db(conn):
    result = conn.execute("""
        SELECT chunk_id, content FROM chunks WHERE type='image' ORDER BY chunk_id
    """).fetchall()

    image_ids = [row[0] for row in result]
    image_texts = [row[1] for row in result]

    print(f"Loaded {len(result)} image chunks")

    return image_ids, image_texts