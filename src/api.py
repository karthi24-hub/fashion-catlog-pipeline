"""
Fashion Visual Search API - Google Lens Style
Multi-item detection with exact color matching and tiered ranking.
"""

from fastapi import FastAPI, UploadFile, Header, HTTPException
import numpy as np
import faiss
import json
import tempfile
import os
import boto3
from dotenv import load_dotenv
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Local imports
from .embed_dinov2 import embed_image
from .clip_classifier import classify_item, get_fallback_categories
from .color_extractor import extract_colors_ensemble
from .filters import apply_all_filters

# YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY not found. Create a .env file with API_KEY=your_key")

app = FastAPI(
    title="Fashion Visual Search API",
    description="Google Lens-style multi-item detection with exact color matching",
    version="2.0.0"
)

# ============ FAISS INDEX LOADING ============
INDEX_PATH = "faiss/catalog.faiss"
ID_MAP_PATH = "faiss/id_map.json"

if not os.path.exists(INDEX_PATH) or not os.path.exists(ID_MAP_PATH):
    logger.warning("FAISS index not found. Will need to build before searching.")
    index = None
    id_map = {}
else:
    index = faiss.read_index(INDEX_PATH)
    with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")

# ============ S3 CONFIG ============
S3_BUCKET = os.getenv("S3_BUCKET", "shoptainment-dev-fashion-dataset-bucket")
S3_PREFIX = os.getenv("S3_PREFIX", "dataset/products/")
S3_REGION = os.getenv("S3_REGION", "ap-south-1")
s3 = boto3.client("s3")

# ============ YOLO MODEL ============
YOLO_WEIGHTS = "models/yolov8n.pt"
yolo_model = None

if YOLO_AVAILABLE:
    if Path(YOLO_WEIGHTS).exists():
        yolo_model = YOLO(YOLO_WEIGHTS)
        logger.info(f"Loaded YOLO model: {YOLO_WEIGHTS}")
    else:
        logger.warning(f"YOLO weights not found at {YOLO_WEIGHTS}")
else:
    logger.warning("Ultralytics not installed. YOLO detection disabled.")

# ============ CONFIGURATION ============
MAX_ITEMS_PER_IMAGE = 5  # Limit items to prevent timeout
FAISS_K = 500  # Number of candidates to retrieve from FAISS
MIN_CONFIDENCE = 0.25  # Minimum YOLO detection confidence
CLIP_CONFIDENCE_THRESHOLD = 0.3  # Minimum CLIP classification confidence


def verify_key(x_api_key: str):
    """Validate API key"""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_meta_from_s3(pid: str) -> dict:
    """Load product meta.json from S3"""
    key = f"{S3_PREFIX}{pid}/meta.json"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Failed to load meta for {pid}: {e}")
        return {}


def get_product_image_url(pid: str, image_name: str = "image_1.jpg") -> str:
    """
    Generate pre-signed S3 URL for product image.
    URL expires after 10 minutes (600 seconds).
    Each search request generates fresh URLs.
    """
    key = f"{S3_PREFIX}{pid}/{image_name}"
    try:
        url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': key
            },
            ExpiresIn=600  # 10 minutes
        )
        return url
    except Exception as e:
        logger.warning(f"Failed to generate presigned URL for {pid}: {e}")
        # Fallback to public URL format (will fail if bucket is private)
        return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


def detect_items_yolo(image_path: str, image: Image.Image) -> List[Dict[str, Any]]:
    """
    Detect clothing items in image using YOLO.
    Falls back to full image if no detections.
    """
    detections = []
    
    if yolo_model is not None:
        try:
            results = yolo_model(image_path, imgsz=1280, conf=MIN_CONFIDENCE, verbose=False)
            
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                
                for i, b in enumerate(boxes):
                    xyxy = b.xyxy.cpu().numpy().tolist()[0]
                    conf = float(b.conf.cpu().numpy()[0])
                    cls_id = int(b.cls.cpu().numpy()[0])
                    
                    # Crop the detected region
                    x1, y1, x2, y2 = map(int, xyxy)
                    w, h = image.size
                    x1 = max(0, min(w-1, x1))
                    x2 = max(x1+1, min(w, x2))
                    y1 = max(0, min(h-1, y1))
                    y2 = max(y1+1, min(h, y2))
                    
                    crop = image.crop((x1, y1, x2, y2))
                    
                    detections.append({
                        "index": i,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "yolo_class": cls_id,
                        "crop": crop
                    })
            
            # Sort by confidence and limit
            detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
            detections = detections[:MAX_ITEMS_PER_IMAGE]
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
    
    # Fallback: use full image if no detections
    if not detections:
        logger.info("No YOLO detections, using full image")
        detections = [{
            "index": 0,
            "bbox": None,
            "confidence": 1.0,
            "yolo_class": None,
            "crop": image
        }]
    
    return detections


def search_faiss(embedding: np.ndarray, k: int = FAISS_K) -> List[Dict[str, Any]]:
    """
    Search FAISS index and load product metadata.
    Returns list of products with scores.
    """
    if index is None:
        return []
    
    # Normalize embedding
    emb = embedding.reshape(1, -1).astype("float32")
    faiss.normalize_L2(emb)
    
    # Search
    D, I = index.search(emb, k)
    
    products = []
    for idx, score in zip(I[0], D[0]):
        pid = id_map.get(str(int(idx)))
        if not pid:
            continue
        
        meta = load_meta_from_s3(pid)
        
        products.append({
            "product_id": pid,
            "similarity_score": float(score),
            "meta": meta
        })
    
    return products


@app.post("/search")
async def search(file: UploadFile, x_api_key: str = Header(None)):
    """
    Multi-item visual search endpoint.
    
    Flow:
    1. YOLO detects items in uploaded image
    2. For each detected item:
       - CLIP classifies category
       - Extract colors
       - DINOv2 embedding
       - Search FAISS (top 500)
       - Post-filter by category + color (tiered)
    3. Return results for all detected items
    """
    verify_key(x_api_key)
    
    if index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")
    
    # Save uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await file.read()
        tmp.write(content)
        image_path = tmp.name
    
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Step 1: YOLO detection
        detections = detect_items_yolo(image_path, image)
        logger.info(f"Detected {len(detections)} items")
        
        detected_items = []
        
        # Step 2: Process each detected item
        for det in detections:
            crop = det["crop"]
            
            # 2a: CLIP classification
            category, clip_conf, specific_label = classify_item(crop, CLIP_CONFIDENCE_THRESHOLD)
            
            # Skip if low confidence and not using fallbacks
            if category == "unknown" and clip_conf < CLIP_CONFIDENCE_THRESHOLD:
                logger.warning(f"Skipping item {det['index']} due to low CLIP confidence: {clip_conf:.2f}")
                continue
            
            # Get fallback categories if needed
            search_categories = get_fallback_categories(category, clip_conf)
            
            # 2b: Extract colors
            # Use empty title for now (could enhance with OCR later)
            detected_colors = extract_colors_ensemble(crop, "")
            
            # 2c: Generate DINOv2 embedding
            # Save crop to temp file (embed_image expects path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_crop:
                crop.save(tmp_crop.name, format="JPEG", quality=90)
                try:
                    embedding = embed_image(tmp_crop.name)
                finally:
                    os.remove(tmp_crop.name)
            
            # 2d: Search FAISS (unified index)
            faiss_results = search_faiss(embedding, k=FAISS_K)
            
            # 2e: Apply post-filters
            filtered_results = apply_all_filters(
                faiss_results,
                category=category,
                colors=detected_colors,
                max_results=10
            )
            
            # 2f: Format matches
            matches = []
            for product in filtered_results:
                pid = product["product_id"]
                meta = product["meta"]
                
                # Get image URL
                images = meta.get("images", [])
                if images:
                    img_path = images[0].get("path", "image_1.jpg")
                else:
                    img_path = "image_1.jpg"
                
                matches.append({
                    "product_id": pid,
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "colors": meta.get("attributes", {}).get("colors", []),
                    "image_url": get_product_image_url(pid, img_path),
                    "buy_link": meta.get("source_url", meta.get("url", "")),
                    "similarity_score": round(product["similarity_score"], 4),
                    "pricing": meta.get("pricing", {})
                })
            
            detected_items.append({
                "item_index": det["index"],
                "bbox": det["bbox"],
                "yolo_confidence": det["confidence"],
                "category": category,
                "specific_label": specific_label,
                "clip_confidence": round(clip_conf, 4),
                "detected_colors": detected_colors[:3],  # Top 3 colors
                "matches": matches
            })
        
        # Handle case: no items detected
        if not detected_items:
            return {
                "detected_items": [],
                "message": "No clothing items detected. Please upload an image with visible clothing."
            }
        
        return {
            "detected_items": detected_items,
            "total_items": len(detected_items)
        }
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        try:
            os.remove(image_path)
        except:
            pass


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "index_loaded": index is not None,
        "index_size": index.ntotal if index else 0,
        "yolo_loaded": yolo_model is not None
    }


@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Fashion Visual Search API",
        "version": "2.0.0",
        "features": [
            "Multi-item detection (YOLO)",
            "Zero-shot classification (CLIP)",
            "Exact color matching (Ensemble)",
            "Tiered similarity ranking"
        ]
    }
