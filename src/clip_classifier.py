"""
CLIP Zero-Shot Classifier for Fashion Items
Uses OpenAI's CLIP model to classify cropped fashion items into categories
without any training. This is the same approach Google Lens uses.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load CLIP model (globally to avoid reloading)
MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading CLIP model: {MODEL_NAME} on {device}")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
logger.info("CLIP model loaded successfully")

# Fashion category labels (hierarchical for better accuracy)
CATEGORY_LABELS = {
    "upperwear": [
        "shirt", "t-shirt", "top", "blouse", "sweater", 
        "jacket", "coat", "hoodie", "cardigan"
    ],
    "lowerwear": [
        "pants", "jeans", "trousers", "shorts", "leggings",
        "skirt", "palazzo"
    ],
    "dress": [
        "dress", "gown", "jumpsuit", "romper", "saree"
    ],
    "footwear": [
        "shoes", "sneakers", "sandals", "boots", "heels",
        "slippers", "loafers"
    ],
    "accessories": [
        "watch", "bracelet", "necklace", "earrings", 
        "ring", "bag", "purse", "handbag", "backpack",
        "belt", "scarf", "hat", "cap", "sunglasses"
    ]
}

# Flatten all labels for CLIP
ALL_LABELS = []
LABEL_TO_CATEGORY = {}
for category, labels in CATEGORY_LABELS.items():
    for label in labels:
        ALL_LABELS.append(label)
        LABEL_TO_CATEGORY[label] = category


def classify_item(image: Image.Image, confidence_threshold: float = 0.3) -> Tuple[str, float, str]:
    """
    Classify a fashion item using CLIP zero-shot learning.
    
    Args:
        image: PIL Image of the cropped fashion item
        confidence_threshold: Minimum confidence to return classification (default: 0.3)
    
    Returns:
        Tuple of (category, confidence, specific_label)
        Example: ("upperwear", 0.87, "shirt")
    
    If confidence < threshold, returns ("unknown", confidence, "unknown")
    """
    try:
        # Prepare inputs for CLIP
        inputs = processor(
            text=ALL_LABELS, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top prediction
        confidence = probs.max().item()
        predicted_idx = probs.argmax().item()
        specific_label = ALL_LABELS[predicted_idx]
        category = LABEL_TO_CATEGORY[specific_label]
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            logger.warning(f"Low confidence classification: {specific_label} ({confidence:.2f})")
            return "unknown", confidence, specific_label
        
        logger.info(f"Classified as: {category}/{specific_label} (confidence: {confidence:.2f})")
        return category, confidence, specific_label
    
    except Exception as e:
        logger.error(f"Error in CLIP classification: {e}")
        return "unknown", 0.0, "unknown"


def classify_batch(images: List[Image.Image], confidence_threshold: float = 0.3) -> List[Tuple[str, float, str]]:
    """
    Classify multiple images in a batch (faster for multi-item detection).
    
    Args:
        images: List of PIL Images
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        List of (category, confidence, specific_label) tuples
    """
    try:
        # Process batch
        inputs = processor(
            text=ALL_LABELS,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        results = []
        for i in range(len(images)):
            confidence = probs[i].max().item()
            predicted_idx = probs[i].argmax().item()
            specific_label = ALL_LABELS[predicted_idx]
            category = LABEL_TO_CATEGORY[specific_label]
            
            if confidence < confidence_threshold:
                results.append(("unknown", confidence, specific_label))
            else:
                results.append((category, confidence, specific_label))
        
        return results
    
    except Exception as e:
        logger.error(f"Error in batch classification: {e}")
        return [("unknown", 0.0, "unknown")] * len(images)


def get_fallback_categories(category: str, confidence: float) -> List[str]:
    """
    Get fallback categories for ambiguous classifications.
    Used when confidence is low (e.g., could be skirt or dress).
    
    Args:
        category: Primary category
        confidence: Classification confidence
    
    Returns:
        List of categories to search (includes fallbacks if confidence < 0.6)
    """
    if confidence >= 0.6:
        return [category]
    
    # Define ambiguous category groups
    AMBIGUOUS_GROUPS = {
        "dress": ["dress", "lowerwear"],  # Could be dress or skirt
        "lowerwear": ["lowerwear", "dress"],  # Could be skirt or dress
        "upperwear": ["upperwear"],  # Generally clear
        "footwear": ["footwear"],  # Generally clear
        "accessories": ["accessories"],  # Generally clear
    }
    
    fallbacks = AMBIGUOUS_GROUPS.get(category, [category])
    logger.info(f"Low confidence ({confidence:.2f}), using fallback categories: {fallbacks}")
    return fallbacks


if __name__ == "__main__":
    # Test the classifier
    import sys
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        img = Image.open(test_path).convert("RGB")
        category, conf, label = classify_item(img)
        print(f"\nClassification Result:")
        print(f"  Category: {category}")
        print(f"  Specific Label: {label}")
        print(f"  Confidence: {conf:.2%}")
        
        if conf < 0.6:
            fallbacks = get_fallback_categories(category, conf)
            print(f"  Fallback Categories: {fallbacks}")
    else:
        print("Usage: python clip_classifier.py <image_path>")
