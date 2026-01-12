import boto3
import os
import io
import numpy as np
from tqdm import tqdm

# your functions
from yolo_detector import detect_and_crop
from embed_dinov2 import embed_image, aggregate_embeddings

# ---------------- CONFIG ----------------
BUCKET = "shoptainment-dev-fashion-dataset-bucket"
PREFIX = "dataset/products/"   # contains product folders
LOCAL_TMP = "tmp_download"
os.makedirs(LOCAL_TMP, exist_ok=True)
# ---------------------------------------

s3 = boto3.client("s3")


def list_product_ids(bucket, prefix):
    """
    Get all product folder names like P000000, P000001 ...
    """
    paginator = s3.get_paginator("list_objects_v2")
    product_ids = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key.replace(prefix, "").split("/")
            if len(parts) >= 2:
                pid = parts[0]
                if pid.startswith("P"):
                    product_ids.add(pid)

    return sorted(product_ids)


def s3_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def download_s3_file(bucket, key, local_path):
    s3.download_file(bucket, key, local_path)


def upload_npy_to_s3(bucket, key, array):
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


# -------------- MAIN PROCESS --------------

product_ids = list_product_ids(BUCKET, PREFIX)
print(" Total products found:", len(product_ids))

for pid in tqdm(product_ids):
    product_prefix = f"{PREFIX}{pid}/"

    embedding_key = f"{product_prefix}embedding.npy"

    # skip if already exists
    if s3_exists(BUCKET, embedding_key):
        continue

    # download image_1 and image_2 if exist
    image_keys = [f"{product_prefix}image_1.jpg", f"{product_prefix}image_2.jpg"]

    embeddings = []

    for img_key in image_keys:
        if not s3_exists(BUCKET, img_key):
            continue

        local_img = os.path.join(LOCAL_TMP, f"{pid}_{os.path.basename(img_key)}")
        download_s3_file(BUCKET, img_key, local_img)

        # YOLO crop (optional: returns crop paths)
        crops = detect_and_crop(local_img)

        for crop_path in crops:
            try:
                emb = embed_image(crop_path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Embedding failed {pid}: {e}")

    if not embeddings:
        print(f" No embeddings for {pid}")
        continue

    final_embedding = aggregate_embeddings(embeddings)
    final_embedding = final_embedding.astype("float32")

    # upload embedding.npy into the same S3 product folder
    upload_npy_to_s3(BUCKET, embedding_key, final_embedding)

print(" Done: embeddings generated and uploaded to S3")
