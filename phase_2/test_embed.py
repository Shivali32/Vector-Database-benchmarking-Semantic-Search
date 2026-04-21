# # test_embed.py
# from transformers import CLIPModel, CLIPProcessor
# import torch
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model.to(device)
# model.eval()

# # Test text
# texts = ["a dog", "a cat"]
# text_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
# with torch.no_grad():
#     text_out = model.text_model(**text_inputs)
#     text_feat = model.text_projection(text_out.pooler_output)
# print(f"text: {text_feat.shape}")  # expect (2, 512)

# # Test image - use any image you have
# img = Image.new("RGB", (224, 224))  # blank image, no file needed
# image_inputs = processor(images=img, return_tensors="pt").to(device)
# with torch.no_grad():
#     image_out = model.vision_model(**image_inputs)
#     image_feat = model.visual_projection(image_out.pooler_output)
# print(f"image: {image_feat.shape}")  # expect (1, 512)

import pandas as pd
# pd.set_option('display.max_colwidth', None)
import duckdb
conn = duckdb.connect("benchmark.duckdb")
# conn = duckdb.connect("chunks.duckdb")

# # print(conn.execute("ALTER TABLE results ADD COLUMN precision_k DOUBLE"))
# print(conn.execute("ALTER TABLE results ADD COLUMN query_embed_time DOUBLE"))
# print(conn.execute("ALTER TABLE results ADD COLUMN answer_embed_time DOUBLE"))
# print(conn.execute("ALTER TABLE results ADD COLUMN avg_scoring_time DOUBLE"))
# conn.execute("ALTER TABLE results ADD COLUMN text_embed_time DOUBLE")
# conn.execute("ALTER TABLE results ADD COLUMN image_embed_time DOUBLE")
# conn.execute("ALTER TABLE results ADD COLUMN index_time DOUBLE")


print("Results Table:")
# print(conn.execute("SELECT * FROM results limit 5").fetchdf())

print(conn.execute("Describe results").fetchdf())
# print("Chunks Table:")
# print(conn.execute("SELECT * FROM chunks Order by doc_id desc limit 5").fetchdf())
# print("Sample Chunks:")
# print(conn.execute("SELECT * FROM chunks Order by chunk_id asc limit 5 offset 1004").fetchdf())


# ids = ['text_4009', 'text_239', 'text_5453']
# ids = ['text_4019', 'text_2319', 'text_5353', 'text_201', 'text_4645', 'text_6211', 'text_193', 'text_4504', 'text_6291', 'text_2399']
# placeholders = ", ".join(["?" for _ in ids])
# rows = conn.execute(f"SELECT chunk_id, content FROM chunks WHERE chunk_id IN ({placeholders})", ids).fetchall()
# print(rows)

# for row in rows:
#     print(row[0], "→", row[1][:100])

# count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
# print("Total rows in chunks table:", count)

# rows = conn.execute("SELECT chunk_id, content FROM chunks Order by chunk_id asc limit 10 offset 7010").fetchdf()
# # rows = conn.execute("SELECT chunk_id, content FROM chunks Order by chunk_id asc limit 10").fetchdf()
# print(rows)

conn.close()
