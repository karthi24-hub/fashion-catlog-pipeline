import os
import json
import boto3
import numpy as np
import faiss
import io

# ---------------- CONFIG ----------------
S3_BUCKET = "shoptainment-dev-fashion-dataset-bucket"
S3_PREFIX = "dataset/products/"
FAISS_DIR = "faiss"
LOCAL_FAISS_PATH = os.path.join(FAISS_DIR, "catalog.faiss")
LOCAL_IDMAP_PATH = os.path.join(FAISS_DIR, "id_map.json")
# ---------------------------------------

os.makedirs(FAISS_DIR, exist_ok=True)

s3 = boto3.client("s3")

def list_embedding_keys(bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("embedding.npy"):
                yield key

def load_npy_from_s3(bucket: str, key: str) -> np.ndarray:
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    arr = np.load(io.BytesIO(data))
    return arr.astype("float32")

embeddings = []
id_map = {}

keys = sorted(list(list_embedding_keys(S3_BUCKET, S3_PREFIX)))

if not keys:
    raise RuntimeError(" No embedding.npy found in S3. Generate embeddings first.")

print(f" Found {len(keys)} embeddings in S3")

for idx, key in enumerate(keys):
    pid = key.split("/")[-2]  # P000000

    emb = load_npy_from_s3(S3_BUCKET, key)

    # handle (512,) or (1,512)
    emb = emb.reshape(1, -1).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(emb)

    embeddings.append(emb[0])
    id_map[str(idx)] = pid

X = np.vstack(embeddings).astype("float32")
dim = X.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(X)

faiss.write_index(index, LOCAL_FAISS_PATH)

with open(LOCAL_IDMAP_PATH, "w", encoding="utf-8") as f:
    json.dump(id_map, f, indent=2)

print(" FAISS index built from S3 embeddings")
print("Saved:", LOCAL_FAISS_PATH)
print("Saved:", LOCAL_IDMAP_PATH)
print("Vectors indexed:", index.ntotal)
print("Embedding dim:", dim)
