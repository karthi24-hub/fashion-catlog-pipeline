"""
Catalog Enrichment Script
Adds category and color attributes to ALL products in S3.
Uses CLIP for classification and ensemble color extraction.

Runtime: ~12-15 hours for 251K products with 4 parallel workers
Run overnight with: python scripts/enrich_catalog.py --workers=4
"""

import boto3
import json
import os
import sys
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.clip_classifier import classify_item
from src.color_extractor import extract_colors_ensemble

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "shoptainment-dev-fashion-dataset-bucket")
S3_PREFIX = os.getenv("S3_PREFIX", "dataset/products/")
s3 = boto3.client("s3")


def list_all_products():
    """List all product IDs from S3"""
    product_ids = []
    paginator = s3.get_paginator("list_objects_v2")
    
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX, Delimiter='/'):
        for prefix in page.get("CommonPrefixes", []):
            # Extract product ID from prefix
            # e.g., "dataset/products/P000001/" -> "P000001"
            pid = prefix['Prefix'].rstrip('/').split('/')[-1]
            product_ids.append(pid)
    
    return product_ids


def load_meta_from_s3(pid: str) -> dict:
    """Load meta.json from S3"""
    key = f"{S3_PREFIX}{pid}/meta.json"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj['Body'].read().decode('utf-8'))
    except Exception as e:
        logger.warning(f"Failed to load meta for {pid}: {e}")
        return None


def load_image_from_s3(pid: str, image_name: str = "image_1.jpg") -> Image.Image:
    """Load product image from S3"""
    key = f"{S3_PREFIX}{pid}/{image_name}"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return Image.open(BytesIO(obj['Body'].read())).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image for {pid}: {e}")
        return None


def save_meta_to_s3(pid: str, meta: dict):
    """Save updated meta.json to S3"""
    key = f"{S3_PREFIX}{pid}/meta.json"
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(meta, indent=2, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save meta for {pid}: {e}")
        return False


def enrich_product(pid: str, force_update: bool = False) -> dict:
    """
    Enrich a single product with category and colors.
    
    Args:
        pid: Product ID
        force_update: If True, re-process even if attributes exist
    
    Returns:
        dict with status and updates made
    """
    result = {
        "product_id": pid,
        "status": "skipped",
        "category_added": False,
        "colors_added": False,
        "error": None
    }
    
    try:
        # Load existing meta
        meta = load_meta_from_s3(pid)
        if meta is None:
            result["status"] = "error"
            result["error"] = "Meta not found"
            return result
        
        # Check if already enriched
        has_category = 'category' in meta and meta['category']
        has_colors = meta.get('attributes', {}).get('colors', [])
        
        if has_category and has_colors and not force_update:
            result["status"] = "already_enriched"
            return result
        
        # Load image for processing
        image = load_image_from_s3(pid)
        if image is None:
            # Fallback: try to extract from title only
            title = meta.get('title', '')
            if title:
                from src.color_extractor import extract_from_title
                colors = extract_from_title(title)
                if colors:
                    meta.setdefault('attributes', {})['colors'] = colors
                    result["colors_added"] = True
            
            if not has_category:
                meta['category'] = 'unknown'
            
            save_meta_to_s3(pid, meta)
            result["status"] = "partial"
            return result
        
        # CLIP classification for category
        if not has_category or force_update:
            category, conf, label = classify_item(image)
            meta['category'] = category
            meta['category_specific'] = label
            meta['category_confidence'] = round(conf, 4)
            result["category_added"] = True
        
        # Color extraction
        if not has_colors or force_update:
            title = meta.get('title', '')
            colors = extract_colors_ensemble(image, title)
            meta.setdefault('attributes', {})['colors'] = colors
            result["colors_added"] = True
        
        # Save updated meta
        if result["category_added"] or result["colors_added"]:
            save_meta_to_s3(pid, meta)
            result["status"] = "enriched"
        else:
            result["status"] = "unchanged"
        
        return result
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"Error enriching {pid}: {e}")
        return result


def enrich_catalog(workers: int = 4, batch_size: int = 1000, force_update: bool = False, 
                   start_from: int = 0, limit: int = None):
    """
    Enrich all products in the catalog.
    
    Args:
        workers: Number of parallel workers
        batch_size: Products to process per batch
        force_update: Re-process even if already enriched
        start_from: Start from this product index (for resuming)
        limit: Max products to process (for testing)
    """
    logger.info("Listing all products from S3...")
    all_products = list_all_products()
    total = len(all_products)
    logger.info(f"Found {total} products")
    
    # Apply start/limit
    products = all_products[start_from:]
    if limit:
        products = products[:limit]
    
    logger.info(f"Processing {len(products)} products (starting from {start_from})")
    
    # Statistics
    stats = {
        "total": len(products),
        "enriched": 0,
        "already_enriched": 0,
        "errors": 0,
        "partial": 0
    }
    
    # Process in batches with progress bar
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(enrich_product, pid, force_update): pid for pid in products}
        
        with tqdm(total=len(products), desc="Enriching catalog") as pbar:
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    result = future.result()
                    
                    if result["status"] == "enriched":
                        stats["enriched"] += 1
                    elif result["status"] == "already_enriched":
                        stats["already_enriched"] += 1
                    elif result["status"] == "error":
                        stats["errors"] += 1
                    elif result["status"] == "partial":
                        stats["partial"] += 1
                    
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Future error for {pid}: {e}")
                
                pbar.update(1)
                
                # Log progress every batch_size products
                if (stats["enriched"] + stats["already_enriched"] + stats["errors"]) % batch_size == 0:
                    logger.info(f"Progress: {stats}")
    
    # Final stats
    logger.info("=" * 50)
    logger.info("Enrichment Complete!")
    logger.info(f"Total processed: {stats['total']}")
    logger.info(f"Newly enriched: {stats['enriched']}")
    logger.info(f"Already enriched: {stats['already_enriched']}")
    logger.info(f"Partial (title only): {stats['partial']}")
    logger.info(f"Errors: {stats['errors']}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich product catalog with category and colors")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for logging")
    parser.add_argument("--force", action="store_true", help="Force re-process already enriched products")
    parser.add_argument("--start-from", type=int, default=0, help="Start from product index (for resuming)")
    parser.add_argument("--limit", type=int, default=None, help="Limit products to process (for testing)")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 10 products")
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running in TEST mode (10 products)")
        args.limit = 10
    
    enrich_catalog(
        workers=args.workers,
        batch_size=args.batch_size,
        force_update=args.force,
        start_from=args.start_from,
        limit=args.limit
    )
