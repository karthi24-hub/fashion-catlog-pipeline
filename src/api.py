from fastapi import FastAPI, UploadFile
import numpy as np
import faiss
import json
import tempfile
import os
from embed_dinov2 import embed_image

app = FastAPI()

index = faiss.read_index("faiss/catalog.faiss")
id_map = json.load(open("faiss/id_map.json"))

@app.post("/search")
async def search(file: UploadFile):
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
            meta_path = os.path.join("dataset", "products", pid, "meta.json")
            meta = {}
            if os.path.exists(meta_path):
                try:
                    meta = json.load(open(meta_path))
                except:
                    meta = {}
            results.append({"product_id": pid, "meta": meta})

        return {"matches": results, "scores": D[0].tolist()}
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
