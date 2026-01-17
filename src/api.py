from fastapi import FastAPI, UploadFile, Header, HTTPException
import numpy as np
import faiss
import json
import tempfile
import os
from dotenv import load_dotenv

from .embed_dinov2 import embed_image

# Load .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY not found. Create a .env file with API_KEY=your_key")

app = FastAPI()

index = faiss.read_index("faiss/catalog.faiss")
id_map = json.load(open("faiss/id_map.json", "r", encoding="utf-8"))

def verify_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/search")
async def search(file: UploadFile, x_api_key: str = Header(None)):
    verify_key(x_api_key)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        emb = embed_image(tmp_path)
        faiss.normalize_L2(emb.reshape(1, -1))

        D, I = index.search(emb.reshape(1, -1), k=5)

        results = []
        for idx in I[0]:
            pid = id_map.get(str(int(idx)))
            if not pid:
                continue

            # local meta.json (same behavior as your old code)
            meta_path = os.path.join("dataset", "products", pid, "meta.json")
            meta = {}

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except:
                    meta = {}

            results.append({"product_id": pid, "meta": meta})

        return {"matches": results, "scores": D[0].tolist()}

    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
