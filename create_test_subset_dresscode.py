#!/usr/bin/env python3
"""
Create a DressCode subset JSON by sampling pairs from the dataset.
"""

import argparse
import json
import os
import random
from pathlib import Path

DRESSCODE_CATEGORIES = ["upper_body", "lower_body", "dresses"]
VALID_SPLITS = ["train", "test"]
VALID_TEST_ORDERS = ["paired", "unpaired"]


def _pairs_filename(split, test_order):
    if split == "train":
        return "train_pairs.txt"
    return f"test_pairs_{test_order}.txt"


def _ensure_image_ext(name):
    base, ext = os.path.splitext(name)
    if ext:
        return name
    return f"{name}.jpg"


def _normalize_pair_path(path):
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _split_category_prefix(path):
    normalized = _normalize_pair_path(path)
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return None, ""
    if parts[0] in DRESSCODE_CATEGORIES:
        category = parts[0]
        rest = parts[1:]
        if rest and rest[0] == "images":
            rest = rest[1:]
        if not rest:
            return category, ""
        return category, "/".join(rest)
    return None, normalized


def _strip_images_prefix(path):
    normalized = _normalize_pair_path(path)
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[0] in ("images", "image"):
        parts = parts[1:]
    return "/".join(parts)


def _load_pairs(
    pairs_path,
    expected_category=None,
    allow_category_prefix=False,
    restrict_categories=None,
    require_category_prefix=False,
):
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"WARNING: Invalid line {line_num} in {pairs_path}: {line}")
                continue

            person_token = parts[0]
            cloth_token = parts[1]

            if allow_category_prefix:
                person_cat, person_path = _split_category_prefix(person_token)
                cloth_cat, cloth_path = _split_category_prefix(cloth_token)
            else:
                person_cat, person_path = None, _normalize_pair_path(person_token)
                cloth_cat, cloth_path = None, _normalize_pair_path(cloth_token)

            if person_cat and cloth_cat and person_cat != cloth_cat:
                raise RuntimeError(
                    f"Category mismatch in {pairs_path} line {line_num}: {person_cat} vs {cloth_cat}"
                )

            line_category = person_cat or cloth_cat
            if line_category is None:
                if require_category_prefix:
                    raise RuntimeError(
                        f"Root-level pairs file lacks category prefix at line {line_num} in {pairs_path}. "
                        "Provide per-category pairs files under each category directory."
                    )
                line_category = expected_category

            if line_category is None:
                raise RuntimeError(
                    f"Unable to determine category for line {line_num} in {pairs_path}."
                )

            if expected_category and line_category != expected_category:
                raise RuntimeError(
                    f"Unexpected category '{line_category}' in {pairs_path} line {line_num}; "
                    f"expected '{expected_category}'."
                )

            if restrict_categories and line_category not in restrict_categories:
                continue

            person_file = _strip_images_prefix(person_path)
            cloth_file = _strip_images_prefix(cloth_path)
            if not person_file or not cloth_file:
                print(f"WARNING: Invalid paths in {pairs_path} line {line_num}: {line}")
                continue

            person_file = _ensure_image_ext(person_file)
            cloth_file = _ensure_image_ext(cloth_file)
            person_file = Path(person_file).name
            cloth_file = Path(cloth_file).name

            pairs.append(
                {
                    "image_file": person_file,
                    "cloth_file": cloth_file,
                    "category": line_category,
                }
            )

    return pairs


def load_dresscode_pairs(root, split, category, test_order):
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split: {split}. Must be one of {VALID_SPLITS}")
    if test_order not in VALID_TEST_ORDERS:
        raise ValueError(f"Invalid test_order: {test_order}. Must be one of {VALID_TEST_ORDERS}")
    if category not in ["all"] + DRESSCODE_CATEGORIES:
        raise ValueError(
            f"Invalid category: {category}. Must be one of ['all'] + {DRESSCODE_CATEGORIES}"
        )
    if not os.path.isdir(root):
        raise FileNotFoundError(f"DressCode root not found: {root}")

    categories = DRESSCODE_CATEGORIES if category == "all" else [category]
    pairs_filename = _pairs_filename(split, test_order)
    data = []
    missing_categories = []

    for cat in categories:
        pairs_path = os.path.join(root, cat, pairs_filename)
        if os.path.exists(pairs_path):
            data.extend(
                _load_pairs(
                    pairs_path,
                    expected_category=cat,
                    allow_category_prefix=True,
                    restrict_categories=None,
                    require_category_prefix=False,
                )
            )
        else:
            missing_categories.append(cat)

    if missing_categories:
        root_pairs_path = os.path.join(root, pairs_filename)
        if not os.path.exists(root_pairs_path):
            missing_str = ", ".join(missing_categories)
            raise FileNotFoundError(
                "Missing pairs for categories: "
                f"{missing_str}. Expected {os.path.join(root, '<category>', pairs_filename)} "
                f"or fallback {root_pairs_path}."
            )

        fallback_pairs = _load_pairs(
            root_pairs_path,
            expected_category=None,
            allow_category_prefix=True,
            restrict_categories=missing_categories,
            require_category_prefix=True,
        )

        if not fallback_pairs:
            missing_str = ", ".join(missing_categories)
            raise RuntimeError(
                "Root-level pairs file did not provide entries for required categories: "
                f"{missing_str}. Provide per-category pairs under {root}."
            )

        loaded_fallback = {item["category"] for item in fallback_pairs}
        still_missing = [cat for cat in missing_categories if cat not in loaded_fallback]
        if still_missing:
            missing_str = ", ".join(still_missing)
            raise RuntimeError(
                "Root-level pairs file missing category prefixes for required categories: "
                f"{missing_str}. Provide per-category pairs under {root}."
            )

        data.extend(fallback_pairs)

    if not data:
        raise RuntimeError("No pairs loaded from DressCode dataset.")

    return data


def create_subset(items, num_samples, seed):
    total = len(items)
    if num_samples <= 0 or total <= num_samples:
        if total <= num_samples:
            print(
                f"WARNING: dataset size ({total}) <= requested ({num_samples}); using all samples."
            )
        return list(items)

    random.seed(seed)
    return random.sample(items, num_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Create a DressCode test subset JSON by sampling pairs."
    )
    parser.add_argument(
        "--dresscode_root",
        type=str,
        default="E:\\BaiduNetdiskDownload\\DressCode",
        help="DressCode root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=VALID_SPLITS,
        help="Dataset split",
    )
    parser.add_argument(
        "--test_order",
        type=str,
        default="paired",
        choices=VALID_TEST_ORDERS,
        help="Test order (only used for split=test)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all"] + DRESSCODE_CATEGORIES,
        help="DressCode category",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="test_subset_500_HD.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples in the subset (<=0 means full set)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("Loading DressCode pairs...")
    items = load_dresscode_pairs(
        root=args.dresscode_root,
        split=args.split,
        category=args.category,
        test_order=args.test_order,
    )
    print(f"Total pairs: {len(items)}")

    subset = create_subset(items, args.num_samples, args.seed)
    print(f"Subset size: {len(subset)}")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    print(f"[OK] Subset saved to: {args.output_json}")
    print("Sample items:")
    for i, item in enumerate(subset[:5]):
        print(f"  {i + 1}. Category: {item.get('category', 'N/A')}")
        print(f"     Person: {item.get('image_file', 'N/A')}")
        print(f"     Cloth: {item.get('cloth_file', 'N/A')}")


if __name__ == "__main__":
    main()
