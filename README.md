Fashion Catalog Pipeline
This repository contains a production-ready fashion visual search pipeline.

Dataset Structure
Each product is stored in its own folder:

dataset/products/P000001/ ├── image_1.jpg ├── image_2.jpg ├── meta.json └── embedding.npy # Generated product embedding

⚙️ Setup
pip install -r requirements.txt

1️ Build product embeddings
Runs YOLO (optional) + DINOv2 and creates one embedding per product.


python src/build_product_embeddings.py
Output:
dataset/products/*/embedding.npy

2️ Build FAISS index
Creates a similarity search index from all product embeddings.

python src/build_faiss_index.py
Output:
faiss/catalog.faiss
faiss/id_map.json

//added requirements.txt

//added color extraction and category classification

python scripts/enrich_catalog.py --workers=4


//updated api

uvicorn src.api:app --host 0.0.0.0 --port 8000