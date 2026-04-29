import json

def load_queries(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def map_passages_to_chunks(queries, chunk_db_conn):
    """
    Maps MS MARCO answer text to actual chunk IDs in your database.
    Finds which chunks contain the ground truth answer.
    """
        # Add to query_loader.py map_passages_to_chunks() function
    for query_item in queries:
        answer_text = query_item.get("answer", "")
        answer_snippet = answer_text[:200]
        
        print(f"\n🔍 Searching for: {answer_snippet[:100]}...")
        
        result = chunk_db_conn.execute("""
            SELECT chunk_id, content FROM chunks 
            WHERE type='text' 
            AND content LIKE ?
            LIMIT 5
        """, (f"%{answer_snippet}%",)).fetchall()
        
        if result:
            print(f"   ✅ Found {len(result)} matching chunks")
        for row in result:
            print(f"      - {row[0]}: {row[1][:100]}...")
    else:
        print(f"   ❌ No matches found")
        
        query_item["passages"] = [row[0] for row in result]
        
        if not query_item["passages"]:
            print(f"Warning: No chunks found for query: {query_item['query'][:50]}")
    
    return queries