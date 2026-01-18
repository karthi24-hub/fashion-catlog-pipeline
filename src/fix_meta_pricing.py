import json
import re
from pathlib import Path
from typing import Optional


DATASET_DIR = Path("dataset/products")   # local path
# Example: dataset/products/P000001/meta.json


def parse_price_to_int(price_value) -> Optional[int]:
    """
    Converts price formats like:
    "Rs. 499" -> 499
    "Rs. 4,999" -> 4999
    499900 -> 4999  (fix if wrongly stored in paise or extra zeros)
    None / "" -> None
    """
    if price_value is None:
        return None

    # if already correct int in range
    if isinstance(price_value, int):
        #  Fix wrong values like 499900, 355900, etc.
        # If divisible by 100 and too large, reduce
        if price_value >= 100000 and price_value % 100 == 0:
            # 499900 -> 4999
            return price_value // 100
        return price_value

    # if string
    if isinstance(price_value, str):
        s = price_value.strip()
        if not s:
            return None

        # remove Rs, INR, commas, spaces
        s = s.replace(",", "")
        s = re.sub(r"(rs\.?|inr|₹)", "", s, flags=re.IGNORECASE).strip()

        # keep only digits
        digits = re.sub(r"[^\d]", "", s)
        if not digits:
            return None

        val = int(digits)

        # Fix if stored in paise style (49900 → 499)
        # But only if it's clearly wrong (too big for fashion price)
        if val >= 100000 and val % 100 == 0:
            return val // 100

        return val

    return None


def fix_meta_file(meta_path: Path) -> bool:
    """
    Fix only pricing.sale and pricing.original in meta.json.
    Returns True if changed.
    """
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f" Failed reading: {meta_path} | {e}")
        return False

    if "pricing" not in meta or not isinstance(meta["pricing"], dict):
        return False

    pricing = meta["pricing"]

    old_sale = pricing.get("sale")
    old_original = pricing.get("original")

    new_sale = parse_price_to_int(old_sale)
    new_original = parse_price_to_int(old_original)

    changed = False

    if new_sale != old_sale:
        pricing["sale"] = new_sale
        changed = True

    if new_original != old_original:
        pricing["original"] = new_original
        changed = True

    if changed:
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f" Failed writing: {meta_path} | {e}")
            return False

    return changed


def main():
    meta_files = list(DATASET_DIR.glob("P*/meta.json"))
    print(f" Found {len(meta_files)} meta.json files")

    changed_count = 0
    for i, meta_path in enumerate(meta_files, 1):
        if fix_meta_file(meta_path):
            changed_count += 1

        if i % 5000 == 0:
            print(f"Processed {i}/{len(meta_files)} | fixed: {changed_count}")

    print("\n==============================")
    print("DONE")
    print(f"Total meta.json files: {len(meta_files)}")
    print(f"Fixed pricing in     : {changed_count}")
    print("==============================")


if __name__ == "__main__":
    main()
