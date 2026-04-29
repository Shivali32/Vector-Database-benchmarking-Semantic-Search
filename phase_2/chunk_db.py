from tqdm import tqdm

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

def insert_text_chunks(conn, text_chunks, batch_size=1000):
    for i in tqdm(range(0, len(text_chunks), batch_size), total=len(text_chunks)//batch_size, desc="Inserting Text Chunks"):
        
        batch = text_chunks[i:i+batch_size]

        data = [
            (
                doc["id"],
                doc["doc_id"],
                "text",
                doc["content"],
                None
            )
            for doc in batch
        ]

        conn.executemany("""
            INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)
        """, data)

    print(f"Inserted {len(text_chunks)} text chunks")


def insert_image_chunks(conn, image_docs, batch_size=500):
    for i in tqdm(range(0, len(image_docs), batch_size), total=len(image_docs)//batch_size, desc="Inserting Image Chunks"):
        
        batch = image_docs[i:i+batch_size]

        data = [
            (
                doc["id"],
                doc["id"],
                "image",
                doc["content"],
                doc["image_path"]
            )
            for doc in batch
        ]

        conn.executemany("""
            INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)
        """, data)

    print(f"Inserted {len(image_docs)} image chunks")

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

def fetch_chunks_by_ids(conn, ids):
    placeholders = ", ".join(["?" for _ in ids])
    rows = conn.execute(
        f"SELECT chunk_id, content FROM chunks WHERE chunk_id IN ({placeholders})",
        ids
    ).fetchall()
    # return in same order as ids
    content_map = {row[0]: row[1] for row in rows}
    return [content_map.get(id_, "") for id_ in ids]