import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DINO_MODEL = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(DEVICE)
DINO_MODEL.eval()

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = DINO_MODEL(tensor)

    return embedding.squeeze().cpu().numpy().astype("float32")

def aggregate_embeddings(embeddings):
    return np.mean(embeddings, axis=0).astype("float32")
