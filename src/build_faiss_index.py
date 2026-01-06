import os
import json
import numpy as np
import faiss

PRODUCTS_DIR = "dataset/products"
FAISS_DIR = "faiss"
os.makedirs(FAISS_DIR, exist_ok=True)

embeddings = []
id_map = {}

product_list = sorted([p for p in os.listdir(PRODUCTS_DIR) if os.path.isdir(os.path.join(PRODUCTS_DIR, p))])

for idx, pid in enumerate(product_list):
    emb_path = os.path.join(PRODUCTS_DIR, pid, "embedding.npy")
    if os.path.exists(emb_path):
        emb = np.load(emb_path).astype("float32")
        # normalize and append normalized vector
        faiss.normalize_L2(emb.reshape(1, -1))
        embeddings.append(emb)
        id_map[str(idx)] = pid

if not embeddings:
    raise RuntimeError("No embeddings found. Run build_product_embeddings first.")

X = np.vstack(embeddings).astype("float32")
dim = X.shape[1]

index = faiss.IndexFlatIP(dim)  # using inner product on normalized vectors = cosine similarity
index.add(X)

faiss.write_index(index, os.path.join(FAISS_DIR, "catalog.faiss"))

with open(os.path.join(FAISS_DIR, "id_map.json"), "w") as f:
    json.dump(id_map, f)

print("✅ FAISS index built")
