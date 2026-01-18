import json
import boto3

BUCKET = "shoptainment-dev-fashion-dataset-bucket"
PREFIX = "dataset/products/"

s3 = boto3.client("s3")


def fix_price(v):
    """
    Fix wrong pricing like 499900 -> 4999
    Only applied if looks like paise mistake.
    """
    if v is None:
        return None

    # ensure int
    try:
        v = int(v)
    except:
        return v

    # main fix rule: 499900 => 4999
    if v >= 100000 and v % 100 == 0:
        return v // 100

    return v


def process_meta(key: str):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = obj["Body"].read().decode("utf-8")
    meta = json.loads(data)

    pricing = meta.get("pricing", {})
    changed = False

    if "sale" in pricing:
        old = pricing["sale"]
        new = fix_price(old)
        if old != new:
            pricing["sale"] = new
            changed = True

    if "original" in pricing and pricing["original"] is not None:
        old = pricing["original"]
        new = fix_price(old)
        if old != new:
            pricing["original"] = new
            changed = True

    if not changed:
        return False

    meta["pricing"] = pricing

    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return True


def main():
    paginator = s3.get_paginator("list_objects_v2")

    total = 0
    updated = 0

    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("meta.json"):
                continue

            total += 1

            try:
                if process_meta(key):
                    updated += 1
                    print(f" Updated: {key}")
            except Exception as e:
                print(f" Failed: {key} -> {e}")

    print("\nDONE ")
    print("Total meta files:", total)
    print("Updated meta files:", updated)


if __name__ == "__main__":
    main()
