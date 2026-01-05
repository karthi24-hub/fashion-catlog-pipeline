import pandas as pd
import requests
import os
import json
import re
import logging
import time
from urllib.parse import urlparse

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_FILE = os.path.join(BASE_DIR, "data", "products.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "products")

IMAGE_COL_PREFIX = "Image"
REQUEST_TIMEOUT = 20
DOWNLOAD_DELAY = 0.3
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def clean_price(value):
    if pd.isna(value):
        return None
    value = re.sub(r"[^\d]", "", str(value))
    return int(value) if value else None

def download_image(url, save_path):
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        logging.error(f"Image failed: {url} | {e}")
        return False

df = pd.read_csv(CSV_FILE)
image_columns = [c for c in df.columns if c.startswith(IMAGE_COL_PREFIX)]

logging.info(f"Found image columns: {image_columns}")

for _, row in df.iterrows():
    product_id = row["product_id"]
    product_dir = os.path.join(OUTPUT_DIR, product_id)
    os.makedirs(product_dir, exist_ok=True)

    images_meta = []
    idx = 1

    for col in image_columns:
        url = row.get(col)
        if pd.isna(url) or not str(url).startswith("http"):
            continue

        ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
        img_name = f"image_{idx}{ext}"
        img_path = os.path.join(product_dir, img_name)

        if download_image(url, img_path):
            images_meta.append({
                "image_id": f"{product_id}_{idx}",
                "path": img_name
            })
            idx += 1

        time.sleep(DOWNLOAD_DELAY)

    meta = {
        "product_id": product_id,
        "title": row.get("Product Title"),
        "category": {
            "id": "earring",
            "label": "Earrings"
        },
        "pricing": {
            "sale": clean_price(row.get("Sale Price")),
            "original": clean_price(row.get("price--original")),
            "currency": "INR"
        },
        "attributes": {
            "colors": [],
            "material": [],
            "gender": "women"
        },
        "images": images_meta,
        "source": {
            "product_page": row.get("Product Detail Page"),
            "brand": "aashirs"
        }
    }

    with open(os.path.join(product_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

logging.info("Dataset build completed.")
