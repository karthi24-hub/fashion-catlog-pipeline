import json
import os
import re
import boto3
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from color_extractor import extract_from_title

# ================= CONFIG =================
# Set to True to update S3, False to update local 'dataset/products'
UPDATE_S3 = True

# Local config
DATASET_DIR = Path("dataset/products")

# S3 config
S3_BUCKET = "shoptainment-dev-fashion-dataset-bucket"
S3_PREFIX = "dataset/products/"

# Mapping keywords to standardized category IDs and Labels
# Order matters: more specific keywords should come first
CATEGORY_RULES = [
    # Jewelry
    (r"earring|stud|dangler|hoop|jhumka|ear\s?top|bali|chandbali", "earring", "Earrings"),
    (r"necklace|pendant|choker|hasli|chain|jewelry\s?set", "necklace", "Necklaces"),
    (r"bracelet|bangle|cuff", "bracelet", "Bracelets & Bangles"),
    (r"ring", "ring", "Rings"),
    
    # Footwear
    (r"flip\s?flop|slide|clog|slipper|adilette", "slippers", "Slippers & Slides"),
    (r"shoes|sneaker|trainer|boot|heel|pumps|flats|footwear|samba|gazelle|superstar|ultraboost|forum|pureboost|adizero|stan\ssmith|nmd|campus|spezial|swift|duramo|supernova|terrex", "shoes", "Shoes"),
    
    # Apparel
    (r"t-shirt|tee|jersey|vest|singlet", "t-shirt", "T-Shirts"),
    (r"polo|shirt", "shirt", "Shirts"),
    (r"hoodie|jacket|zip|coat|sweater|cardigan|pullover|blazer|windbreaker|fleece|outerwear", "outerwear", "Outerwear"),
    (r"dress|gown|frock|saree|sari|jumpsuit|romper", "dress", "Dresses"),
    (r"shorts|skort", "shorts", "Shorts"),
    (r"skirt", "skirt", "Skirts"),
    (r"pants|trousers|leggings|track\s?pants|joggers|sweatpants", "pants", "Pants & Trousers"),
    
    # Accessories
    (r"socks", "socks", "Socks"),
    (r"bag|backpack|tote|waist\s?bag|purse|handbag", "bag", "Bags"),
    (r"cap|hat|beanie|visor|headband", "headwear", "Headwear"),
    (r"watch", "watch", "Watches"),
    (r"sunglasses|eyewear", "eyewear", "Eyewear"),
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def classify_by_title(title: str) -> Dict[str, str]:
    """Identify category based on title keywords."""
    if not title:
        return {"id": "apparel", "label": "Fashion Items"}
        
    title_lower = title.lower()
    for pattern, cat_id, label in CATEGORY_RULES:
        if re.search(pattern, title_lower):
            return {"id": cat_id, "label": label}
            
    return {"id": "apparel", "label": "Fashion Items"}

def get_gender(title: str) -> str:
    """Identify gender based on title keywords."""
    title_lower = title.lower()
    if any(w in title_lower for w in ["women", "lady", "girls", "female"]):
        return "women"
    if any(w in title_lower for w in ["men", "boys", "male", "unisex"]):
        return "unisex" if "unisex" in title_lower else "men"
    return "unisex"

def process_meta(meta: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Update metadata fields. Returns (updated_meta, changed)."""
    changed = False
    title = meta.get("title", "")
    
    # 1. Update Category
    current_cat = meta.get("category", {})
    # If it's a string, or if it's the wrong label
    new_cat = classify_by_title(title)
    
    if isinstance(current_cat, str) or current_cat.get("id") != new_cat["id"]:
        meta["category"] = new_cat
        changed = True
        
    # 2. Update Colors
    if "attributes" not in meta:
        meta["attributes"] = {}
    
    attrs = meta["attributes"]
    old_colors = attrs.get("colors", [])
    
    # Use the color_extractor logic
    new_colors = extract_from_title(title)
    
    if new_colors != old_colors:
        attrs["colors"] = new_colors
        changed = True
        
    # 3. Update Gender
    old_gender = attrs.get("gender")
    new_gender = get_gender(title)
    if old_gender != new_gender:
        attrs["gender"] = new_gender
        changed = True
        
    return meta, changed

# ============ LOCAL PROCESSING ============

def fix_local():
    meta_files = list(DATASET_DIR.glob("P*/meta.json"))
    logger.info(f"Found {len(meta_files)} local meta.json files")
    
    changed_count = 0
    for i, path in enumerate(meta_files, 1):
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            updated_meta, changed = process_meta(meta)
            
            if changed:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(updated_meta, f, indent=2, ensure_ascii=False)
                changed_count += 1
                
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(meta_files)} | Updated: {changed_count}")
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            
    logger.info(f"Finished. Total: {len(meta_files)}, Updated: {changed_count}")

# ============ S3 PROCESSING ============

def fix_s3():
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    
    logger.info(f"Listing meta.json files in s3://{S3_BUCKET}/{S3_PREFIX}")
    
    meta_keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("meta.json"):
                meta_keys.append(obj["Key"])
                
    logger.info(f"Found {len(meta_keys)} meta.json files in S3")
    
    def process_s3_key(key):
        try:
            # Download
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            meta = json.loads(obj["Body"].read().decode("utf-8"))
            
            # Update
            updated_meta, changed = process_meta(meta)
            
            # Upload if changed
            if changed:
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=key,
                    Body=json.dumps(updated_meta, indent=2, ensure_ascii=False).encode("utf-8")
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Error on S3 key {key}: {e}")
            return False

    updated_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(process_s3_key, key): key for key in meta_keys}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            if future.result():
                updated_count += 1
            if i % 500 == 0:
                logger.info(f"Progress: {i}/{len(meta_keys)} | Updated: {updated_count}")

    logger.info(f"S3 fix complete. Updated {updated_count} files.")

if __name__ == "__main__":
    if UPDATE_S3:
        fix_s3()
    else:
        # Check if local dir exists, if not maybe the user is in a different CWD
        if not DATASET_DIR.exists():
            logger.error(f"Local directory {DATASET_DIR} not found. Ensure you are in the project root.")
        else:
            fix_local()
