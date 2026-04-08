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

    for doc in text_chunks:
        data.append((
            doc["id"],
            doc["doc_id"],
            "text",
            doc["content"],
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
        SELECT chunk_id, doc_id, content FROM chunks WHERE type='text'
    """).fetchall()

    print(f"Loaded {len(result)} text chunks")
    
    text_docs = [
        {
            "id": row[0],
            "doc_id": row[1],
            "content": row[2]
        }
        for row in result
    ]

    return text_docs


def load_image_chunks_from_db(conn):
    result = conn.execute("""
        SELECT chunk_id, content, image_path 
        FROM chunks 
        WHERE type='image' 
        ORDER BY chunk_id
    """).fetchall()

    image_docs = []

    for row in result:
        image_docs.append({
            "id": row[0],
            "content": row[1],
            "image_path": row[2]
        })

    print(f"Loaded {len(result)} image chunks")

    return image_docs