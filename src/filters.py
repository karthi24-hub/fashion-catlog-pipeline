"""
Post-Filtering Module for Fashion Search
Implements Google Lens-style filtering: Category + Tiered Color Matching
Filters are applied AFTER FAISS search on the unified index.
"""

from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Category groupings for flexible matching
CATEGORY_GROUPS = {
    'shirt': ['shirt', 't-shirt', 'top', 'blouse', 'polo'],
    't-shirt': ['shirt', 't-shirt', 'top'],
    'top': ['shirt', 't-shirt', 'top', 'blouse'],
    'blouse': ['blouse', 'top', 'shirt'],
    'sweater': ['sweater', 'cardigan', 'pullover', 'hoodie'],
    'jacket': ['jacket', 'coat', 'blazer', 'hoodie'],
    'hoodie': ['hoodie', 'sweater', 'jacket'],
    
    'pants': ['pants', 'trousers', 'jeans', 'chinos'],
    'jeans': ['jeans', 'pants', 'denim'],
    'trousers': ['trousers', 'pants', 'chinos'],
    'shorts': ['shorts'],
    'leggings': ['leggings', 'tights'],
    'skirt': ['skirt'],
    
    'dress': ['dress', 'gown', 'frock'],
    'gown': ['gown', 'dress'],
    'jumpsuit': ['jumpsuit', 'romper'],
    'saree': ['saree', 'sari'],
    
    'shoes': ['shoes', 'sneakers', 'footwear'],
    'sneakers': ['sneakers', 'shoes', 'trainers'],
    'sandals': ['sandals', 'slippers', 'flats'],
    'boots': ['boots', 'shoes'],
    'heels': ['heels', 'pumps', 'shoes'],
    
    'watch': ['watch', 'smartwatch'],
    'bracelet': ['bracelet', 'bangle'],
    'necklace': ['necklace', 'chain', 'pendant'],
    'earrings': ['earrings', 'studs'],
    'ring': ['ring'],
    'bag': ['bag', 'purse', 'handbag', 'tote'],
    'backpack': ['backpack', 'bag'],
    
    # Parent categories
    'upperwear': ['shirt', 't-shirt', 'top', 'blouse', 'sweater', 'jacket', 'hoodie', 'cardigan', 'coat'],
    'lowerwear': ['pants', 'jeans', 'trousers', 'shorts', 'leggings', 'skirt', 'palazzo'],
    'footwear': ['shoes', 'sneakers', 'sandals', 'boots', 'heels', 'slippers', 'loafers'],
    'accessories': ['watch', 'bracelet', 'necklace', 'earrings', 'ring', 'bag', 'purse', 'handbag', 
                    'backpack', 'belt', 'scarf', 'hat', 'cap', 'sunglasses'],
}

# Color similarity tiers for tiered ranking
COLOR_SIMILARITY = {
    'red': {'exact': ['red'], 'similar': ['crimson', 'scarlet', 'cherry'], 'related': ['maroon', 'pink', 'orange']},
    'maroon': {'exact': ['maroon'], 'similar': ['burgundy', 'wine', 'bordeaux'], 'related': ['red', 'brown', 'purple']},
    'pink': {'exact': ['pink'], 'similar': ['rose', 'blush', 'coral'], 'related': ['red', 'purple', 'peach']},
    'blue': {'exact': ['blue'], 'similar': ['azure', 'cobalt', 'royal'], 'related': ['navy', 'teal', 'purple']},
    'navy': {'exact': ['navy'], 'similar': ['midnight', 'dark blue'], 'related': ['blue', 'black']},
    'teal': {'exact': ['teal'], 'similar': ['turquoise', 'cyan', 'aqua'], 'related': ['blue', 'green']},
    'green': {'exact': ['green'], 'similar': ['emerald', 'lime', 'mint'], 'related': ['olive', 'teal', 'khaki']},
    'olive': {'exact': ['olive'], 'similar': ['army', 'military'], 'related': ['green', 'brown', 'khaki']},
    'yellow': {'exact': ['yellow'], 'similar': ['gold', 'golden', 'amber'], 'related': ['orange', 'beige', 'mustard']},
    'orange': {'exact': ['orange'], 'similar': ['tangerine', 'rust'], 'related': ['red', 'yellow', 'brown', 'peach']},
    'white': {'exact': ['white'], 'similar': ['ivory', 'cream', 'off-white'], 'related': ['beige', 'grey', 'silver']},
    'black': {'exact': ['black'], 'similar': ['ebony', 'jet', 'onyx'], 'related': ['grey', 'navy', 'charcoal']},
    'grey': {'exact': ['grey', 'gray'], 'similar': ['charcoal', 'slate', 'silver', 'ash'], 'related': ['black', 'white']},
    'brown': {'exact': ['brown'], 'similar': ['chocolate', 'coffee', 'tan'], 'related': ['beige', 'orange', 'maroon', 'khaki']},
    'beige': {'exact': ['beige'], 'similar': ['tan', 'nude', 'cream', 'camel'], 'related': ['brown', 'white', 'khaki']},
    'purple': {'exact': ['purple'], 'similar': ['violet', 'lavender', 'plum'], 'related': ['pink', 'blue', 'maroon']},
    'gold': {'exact': ['gold'], 'similar': ['golden', 'brass'], 'related': ['yellow', 'brown', 'orange']},
    'silver': {'exact': ['silver'], 'similar': ['chrome', 'metallic'], 'related': ['grey', 'white']},
}


def filter_by_category(products: List[Dict[str, Any]], target_category: str) -> List[Dict[str, Any]]:
    """
    Filter products to match the target category.
    Uses flexible matching - e.g., "shirt" matches "t-shirt", "top", etc.
    
    Args:
        products: List of product dictionaries with 'meta' containing 'category'
        target_category: Category to filter for (e.g., "shirt", "upperwear")
    
    Returns:
        Filtered list of products matching the category
    """
    if not target_category or target_category == "unknown":
        return products  # No filtering if category unknown
    
    # Get allowed categories for matching
    allowed = CATEGORY_GROUPS.get(target_category.lower(), [target_category.lower()])
    
    filtered = []
    for product in products:
        product_category = product.get('meta', {}).get('category', '').lower()
        
        # Direct match or group match
        if product_category in allowed or product_category == target_category.lower():
            filtered.append(product)
    
    logger.info(f"Category filter: {len(products)} → {len(filtered)} products (target: {target_category})")
    return filtered


def get_color_tier(query_color: str, product_color: str) -> str:
    """
    Determine the similarity tier between query and product colors.
    
    Returns: "exact", "similar", "related", or "none"
    """
    qc = query_color.lower()
    pc = product_color.lower()
    
    # Exact match
    if qc == pc:
        return "exact"
    
    # Check similarity tiers
    sim_data = COLOR_SIMILARITY.get(qc, {})
    
    if pc in sim_data.get('exact', []):
        return "exact"
    if pc in sim_data.get('similar', []):
        return "similar"
    if pc in sim_data.get('related', []):
        return "related"
    
    return "none"


def filter_by_color_tiered(
    products: List[Dict[str, Any]], 
    query_colors: List[str],
    include_no_color: bool = True
) -> List[Dict[str, Any]]:
    """
    Filter and rank products by color using tiered matching.
    Google Lens approach: Exact matches first → Similar → Related
    
    Args:
        products: List of product dictionaries
        query_colors: List of colors to match (e.g., ["maroon", "white"])
        include_no_color: Whether to include products with no color data (appended at end)
    
    Returns:
        Products sorted by color match tier
    """
    if not query_colors:
        return products  # No color filtering if no colors detected
    
    exact_matches = []
    similar_matches = []
    related_matches = []
    no_color_matches = []
    
    for product in products:
        # Get product colors from meta
        product_colors = product.get('meta', {}).get('attributes', {}).get('colors', [])
        
        # Handle products with no color data
        if not product_colors:
            if include_no_color:
                no_color_matches.append(product)
            continue
        
        # Normalize product colors
        product_colors = [c.lower() for c in product_colors]
        
        # Find best matching tier across all query colors
        best_tier = "none"
        
        for qc in query_colors:
            for pc in product_colors:
                tier = get_color_tier(qc, pc)
                
                # Update best tier (exact > similar > related > none)
                if tier == "exact":
                    best_tier = "exact"
                    break
                elif tier == "similar" and best_tier != "exact":
                    best_tier = "similar"
                elif tier == "related" and best_tier not in ["exact", "similar"]:
                    best_tier = "related"
            
            if best_tier == "exact":
                break  # Found exact match, no need to continue
        
        # Categorize by tier
        if best_tier == "exact":
            exact_matches.append(product)
        elif best_tier == "similar":
            similar_matches.append(product)
        elif best_tier == "related":
            related_matches.append(product)
    
    # Combine in tiered order
    result = exact_matches + similar_matches + related_matches
    
    if include_no_color:
        result += no_color_matches
    
    logger.info(f"Color filter: {len(exact_matches)} exact, {len(similar_matches)} similar, "
                f"{len(related_matches)} related, {len(no_color_matches)} no-color")
    
    return result


def filter_by_gender(products: List[Dict[str, Any]], target_gender: str) -> List[Dict[str, Any]]:
    """
    Optional: Filter products by gender.
    
    Args:
        products: List of product dictionaries
        target_gender: "men", "women", "unisex", or None
    
    Returns:
        Filtered products
    """
    if not target_gender:
        return products
    
    target_gender = target_gender.lower()
    
    filtered = []
    for product in products:
        product_gender = product.get('meta', {}).get('gender', '').lower()
        
        # Match target gender or unisex products
        if product_gender == target_gender or product_gender == 'unisex' or not product_gender:
            filtered.append(product)
    
    logger.info(f"Gender filter: {len(products)} → {len(filtered)} products (target: {target_gender})")
    return filtered


def apply_all_filters(
    products: List[Dict[str, Any]],
    category: str = None,
    colors: List[str] = None,
    gender: str = None,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Apply all post-filters in sequence.
    Order: Category → Color (tiered) → Gender → Limit
    
    Args:
        products: Raw FAISS search results
        category: Target category
        colors: Query colors
        gender: Target gender (optional)
        max_results: Maximum results to return
    
    Returns:
        Filtered and ranked products
    """
    result = products
    
    # Apply category filter
    if category:
        result = filter_by_category(result, category)
    
    # Apply tiered color filter
    if colors:
        result = filter_by_color_tiered(result, colors)
    
    # Apply gender filter (optional)
    if gender:
        result = filter_by_gender(result, gender)
    
    # Limit results
    final_result = result[:max_results]
    
    logger.info(f"Final filtering: {len(products)} → {len(final_result)} products")
    return final_result


if __name__ == "__main__":
    # Test the filters
    test_products = [
        {"product_id": "P001", "meta": {"category": "shirt", "attributes": {"colors": ["maroon"]}}},
        {"product_id": "P002", "meta": {"category": "shirt", "attributes": {"colors": ["burgundy"]}}},
        {"product_id": "P003", "meta": {"category": "shirt", "attributes": {"colors": ["blue"]}}},
        {"product_id": "P004", "meta": {"category": "pants", "attributes": {"colors": ["red"]}}},
        {"product_id": "P005", "meta": {"category": "t-shirt", "attributes": {"colors": ["red"]}}},
    ]
    
    print("\n=== Filter Test ===")
    print(f"Original: {len(test_products)} products")
    
    # Filter by category
    filtered = filter_by_category(test_products, "shirt")
    print(f"After category filter (shirt): {[p['product_id'] for p in filtered]}")
    
    # Filter by color
    filtered = filter_by_color_tiered(test_products, ["maroon"])
    print(f"After color filter (maroon): {[p['product_id'] for p in filtered]}")
    
    # Combined
    final = apply_all_filters(test_products, category="shirt", colors=["maroon"])
    print(f"Combined filter: {[p['product_id'] for p in final]}")
