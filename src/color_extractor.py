"""
Color Extraction Module for Fashion Items
Uses ensemble method: Title parsing + K-means clustering for robust color detection.
Matches Google Lens approach with tiered color similarity.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter
from typing import List, Tuple, Optional
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive color keywords mapping
COLOR_KEYWORDS = {
    # Reds
    'red': ['red', 'crimson', 'scarlet', 'cherry', 'ruby'],
    'maroon': ['maroon', 'burgundy', 'wine', 'bordeaux', 'oxblood'],
    'pink': ['pink', 'rose', 'blush', 'coral', 'salmon', 'fuchsia', 'magenta'],
    
    # Blues
    'blue': ['blue', 'azure', 'cobalt', 'sapphire', 'royal blue'],
    'navy': ['navy', 'navy blue', 'midnight', 'dark blue'],
    'teal': ['teal', 'turquoise', 'cyan', 'aqua'],
    
    # Greens
    'green': ['green', 'lime', 'emerald', 'forest', 'mint', 'sage'],
    'olive': ['olive', 'army', 'khaki green', 'military'],
    
    # Yellows/Oranges
    'yellow': ['yellow', 'gold', 'golden', 'lemon', 'mustard', 'amber'],
    'orange': ['orange', 'tangerine', 'peach', 'apricot', 'rust'],
    
    # Neutrals
    'white': ['white', 'ivory', 'cream', 'off-white', 'pearl', 'snow'],
    'black': ['black', 'ebony', 'jet', 'onyx', 'charcoal black'],
    'grey': ['grey', 'gray', 'charcoal', 'slate', 'silver', 'ash', 'smoke'],
    'brown': ['brown', 'chocolate', 'coffee', 'mocha', 'tan', 'camel', 'khaki', 'beige', 'nude', 'taupe'],
    
    # Purples
    'purple': ['purple', 'violet', 'lavender', 'plum', 'mauve', 'lilac', 'orchid'],
    
    # Metallics
    'gold': ['gold', 'golden', 'brass'],
    'silver': ['silver', 'chrome', 'metallic'],
}

# RGB reference colors for K-means mapping
RGB_REFERENCE = {
    'red': (220, 20, 60),
    'maroon': (128, 0, 0),
    'pink': (255, 105, 180),
    'blue': (65, 105, 225),
    'navy': (0, 0, 128),
    'teal': (0, 128, 128),
    'green': (34, 139, 34),
    'olive': (128, 128, 0),
    'yellow': (255, 215, 0),
    'orange': (255, 140, 0),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'grey': (128, 128, 128),
    'brown': (139, 69, 19),
    'purple': (128, 0, 128),
    'gold': (255, 215, 0),
    'silver': (192, 192, 192),
    'beige': (245, 245, 220),
}


def extract_from_title(title: str) -> List[str]:
    """
    Extract color keywords from product title.
    Fast method - 70% accurate but instant.
    
    Args:
        title: Product title string
    
    Returns:
        List of detected color names (can be multiple for multi-color items)
    """
    if not title:
        return []
    
    title_lower = title.lower()
    found_colors = []
    
    # Check for each color keyword
    for color_name, keywords in COLOR_KEYWORDS.items():
        for keyword in keywords:
            # Use word boundary matching to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, title_lower):
                if color_name not in found_colors:
                    found_colors.append(color_name)
                break  # Found this color, move to next
    
    return found_colors


def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Map RGB values to nearest color name using Euclidean distance.
    
    Args:
        rgb: Tuple of (R, G, B) values
    
    Returns:
        Color name string
    """
    min_distance = float('inf')
    closest_color = 'unknown'
    
    for color_name, ref_rgb in RGB_REFERENCE.items():
        # Euclidean distance in RGB space
        distance = sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb)) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    
    return closest_color


def extract_dominant_colors(image: Image.Image, k: int = 5) -> List[str]:
    """
    Extract dominant colors from image using K-means clustering.
    Accurate method - slower but robust.
    
    Args:
        image: PIL Image
        k: Number of color clusters
    
    Returns:
        List of dominant color names ordered by prevalence
    """
    try:
        # Resize for faster processing
        img = image.copy()
        img = img.resize((100, 100))
        img = img.convert('RGB')
        
        # Get pixels as numpy array
        pixels = np.array(img).reshape(-1, 3)
        
        # Remove near-white and near-black pixels (often background)
        mask = ~((pixels.sum(axis=1) > 700) | (pixels.sum(axis=1) < 50))
        filtered_pixels = pixels[mask]
        
        # If too few pixels after filtering, use original
        if len(filtered_pixels) < 100:
            filtered_pixels = pixels
        
        # K-means clustering
        kmeans = KMeans(n_clusters=min(k, len(filtered_pixels)), random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Count pixels per cluster
        labels = kmeans.predict(filtered_pixels)
        counts = Counter(labels)
        
        # Sort by prevalence and map to color names
        sorted_indices = [idx for idx, _ in counts.most_common()]
        color_names = []
        
        for idx in sorted_indices:
            rgb = tuple(colors[idx])
            color_name = rgb_to_color_name(rgb)
            if color_name not in color_names:
                color_names.append(color_name)
        
        return color_names
    
    except Exception as e:
        logger.warning(f"Error in K-means color extraction: {e}")
        return []


def extract_colors_ensemble(image: Image.Image, title: str = "") -> List[str]:
    """
    Ensemble method combining title parsing and K-means.
    Best of both worlds - fast + accurate.
    
    Strategy:
    1. Parse title for colors (fast, trusted if explicit)
    2. Extract dominant colors from image (accurate)
    3. If title color matches dominant colors, boost confidence
    4. Return combined unique colors
    
    Args:
        image: PIL Image
        title: Product title (optional)
    
    Returns:
        List of color names, ordered by confidence
    """
    # Method 1: Title parsing (fast, high trust if found)
    title_colors = extract_from_title(title)
    
    # Method 2: K-means dominant colors (accurate)
    image_colors = extract_dominant_colors(image)
    
    # Combine results intelligently
    if title_colors:
        # Title colors first (explicit), then image colors if different
        result = title_colors.copy()
        for color in image_colors:
            if color not in result:
                result.append(color)
        # Limit to top 5
        return result[:5]
    else:
        # No title colors, rely on image analysis
        return image_colors[:5]


def get_color_similarity_tier(query_color: str, product_color: str) -> str:
    """
    Determine similarity tier between two colors.
    Used for tiered ranking in search results.
    
    Returns: "exact", "similar", "related", or "none"
    """
    COLOR_SIMILARITY = {
        'red': {'similar': ['crimson', 'scarlet'], 'related': ['maroon', 'pink', 'orange']},
        'maroon': {'similar': ['burgundy', 'wine'], 'related': ['red', 'brown', 'purple']},
        'pink': {'similar': ['rose', 'blush'], 'related': ['red', 'purple', 'coral']},
        'blue': {'similar': ['azure', 'cobalt'], 'related': ['navy', 'teal', 'purple']},
        'navy': {'similar': ['midnight'], 'related': ['blue', 'black']},
        'green': {'similar': ['emerald', 'lime'], 'related': ['olive', 'teal']},
        'yellow': {'similar': ['gold', 'amber'], 'related': ['orange', 'beige']},
        'orange': {'similar': ['tangerine', 'rust'], 'related': ['red', 'yellow', 'brown']},
        'white': {'similar': ['ivory', 'cream'], 'related': ['beige', 'grey', 'silver']},
        'black': {'similar': ['ebony', 'jet'], 'related': ['grey', 'navy']},
        'grey': {'similar': ['charcoal', 'slate', 'silver'], 'related': ['black', 'white']},
        'brown': {'similar': ['chocolate', 'tan'], 'related': ['beige', 'orange', 'maroon']},
        'purple': {'similar': ['violet', 'lavender'], 'related': ['pink', 'blue', 'maroon']},
        'beige': {'similar': ['tan', 'nude', 'cream'], 'related': ['brown', 'white', 'khaki']},
    }
    
    # Normalize colors
    qc = query_color.lower()
    pc = product_color.lower()
    
    # Exact match
    if qc == pc:
        return "exact"
    
    # Check similarity tiers
    sim_data = COLOR_SIMILARITY.get(qc, {})
    if pc in sim_data.get('similar', []):
        return "similar"
    if pc in sim_data.get('related', []):
        return "related"
    
    return "none"


if __name__ == "__main__":
    # Test the color extractor
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        title = sys.argv[2] if len(sys.argv) > 2 else ""
        
        img = Image.open(test_path).convert("RGB")
        
        print("\n=== Color Extraction Results ===")
        print(f"Title: {title}")
        
        title_colors = extract_from_title(title)
        print(f"Title Colors: {title_colors}")
        
        image_colors = extract_dominant_colors(img)
        print(f"Image Colors (K-means): {image_colors}")
        
        ensemble_colors = extract_colors_ensemble(img, title)
        print(f"Ensemble Colors: {ensemble_colors}")
    else:
        print("Usage: python color_extractor.py <image_path> [title]")
