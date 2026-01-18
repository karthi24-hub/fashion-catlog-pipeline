from fastapi import FastAPI, UploadFile, Header, HTTPException
import numpy as np
import faiss
import json
import tempfile
import os
import boto3
from dotenv import load_dotenv

from .embed_dinov2 import embed_image

# Load .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY not found. Create a .env file with API_KEY=your_key")

app = FastAPI()

# Load FAISS index + id map (LOCAL)
index = faiss.read_index("faiss/catalog.faiss")
with open("faiss/id_map.json", "r", encoding="utf-8") as f:
    id_map = json.load(f)

# -------- S3 CONFIG --------
S3_BUCKET = "shoptainment-dev-fashion-dataset-bucket"
S3_PREFIX = "dataset/products/"
S3_REGION = "ap-south-1"   #  add region for correct url
s3 = boto3.client("s3")


def verify_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_meta_from_s3(pid: str) -> dict:
    """
    Load meta.json from S3:
    s3://bucket/dataset/products/{pid}/meta.json
    """
    key = f"{S3_PREFIX}{pid}/meta.json"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return {}


def get_product_image_url(pid: str) -> str:
    """
    Always return image_1.jpg public url from s3.
    """
    key = f"{S3_PREFIX}{pid}/image_1.jpg"
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


@app.post("/search")
async def search(file: UploadFile, x_api_key: str = Header(None)):
    verify_key(x_api_key)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Embed query image
        emb = embed_image(tmp_path)
        faiss.normalize_L2(emb.reshape(1, -1))

        # Search
        D, I = index.search(emb.reshape(1, -1), k=5)

        results = []
        for idx in I[0]:
            pid = id_map.get(str(int(idx)))
            if not pid:
                continue

            #  Load meta from S3
            meta = load_meta_from_s3(pid)

            #  Add 1 image url
            image_url = get_product_image_url(pid)

            results.append({
                "product_id": pid,
                "meta": meta,
                "image_url": image_url
            })

        return {"matches": results, "scores": D[0].tolist()}

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
