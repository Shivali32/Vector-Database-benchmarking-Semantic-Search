import os, re
import json

def load_documents(data_path):
    documents = []
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                raw_text = f.read()
                clean_text = clean_wiki_text(raw_text)

                documents.append({
                    "doc_id": os.path.splitext(file)[0],
                    "content": clean_text
                })
    return documents

def chunk_text(text, chunk_size=500, overlap=50):
    
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    # chunks = []
    # start = 0

    # while start < len(text):
    #     end = start + chunk_size
    #     chunk = text[start:end]
    #     if len(chunk) >= 50:
    #         chunks.append(chunk)
    #     start += chunk_size - overlap

    # # print(chunks[:5])
    
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
        raw_path = record.get("local_path", "")
        clean_path = raw_path.replace("\\", "/")  
        filename = os.path.basename(clean_path)   
        image_path = os.path.join("wit_images", filename)
        image_path = os.path.normpath(image_path)

        documents.append({
            "id": record["image_id"] or "",
            "type": "image",
            "content": record.get("caption") or "",
            "image_path": image_path or "",
            "metadata": {
                "page_title": record.get("page_title") or "",
                "source": "wit"
            }
        })

    return documents


def clean_wiki_text(text):
    trigger_sections = [
        "references",
        "external links",
        "see also",
        "further reading",
        "sources",
        "bibliography"
    ]

    text_lower = text.lower()

    cut_index = len(text)

    for trigger in trigger_sections:
        idx = text_lower.find(trigger)
        if idx != -1:
            cut_index = min(cut_index, idx)

    return text[:cut_index]    