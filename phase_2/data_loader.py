import os
import json

def load_documents(data_path):
    documents = []
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_wit_images(metadata_path):
    """
    Loads WIT metadata and prepares image documents
    in same format as text docs.
    """

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"{metadata_path} not found")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    documents = []

    for record in metadata:
        documents.append({
            "id": record["image_id"],
            "type": "image",
            "content": record.get("caption", ""),
            "image_path": record.get("local_path"),
            "metadata": {
                "page_title": record.get("page_title"),
                "source": "wit"
            }
        })

    return documents