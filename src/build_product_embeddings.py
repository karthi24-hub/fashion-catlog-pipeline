import os
import json
import numpy as np
from yolo_detector import detect_and_crop
from embed_dinov2 import embed_image, aggregate_embeddings

PRODUCTS_DIR = "dataset/products"

for product_id in os.listdir(PRODUCTS_DIR):
    product_dir = os.path.join(PRODUCTS_DIR, product_id)
    if not os.path.isdir(product_dir):
        continue

    emb_file = os.path.join(product_dir, "embedding.npy")
    if os.path.exists(emb_file):
        print(f"Skipping {product_id} (embedding exists)")
        continue

    crops_dir = os.path.join(product_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    embeddings = []

    for file in os.listdir(product_dir):
        if file.lower().endswith(".jpg"):
            image_path = os.path.join(product_dir, file)
            crops = detect_and_crop(image_path, crops_dir)
            for crop in crops:
                try:
                    emb = embed_image(crop)
                    embeddings.append(emb)
                except Exception as e:
                    print(f"Embedding failed for {crop}: {e}")

    if not embeddings:
        print(f"No embeddings for {product_id}, skipping")
        continue

    final_embedding = aggregate_embeddings(embeddings)
    np.save(emb_file, final_embedding)
    print(f"✅ Embedded {product_id}")
