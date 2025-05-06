#!/usr/bin/env python3

"""
remove_supercategory.py

Description:
    This script removes the supercategory "supermarket-product" from the COCO-style
    dataset annotations. It updates the category IDs and saves the cleaned
    annotations to a new file.

Usage:
    Remove the supercategory using the newest dataset version:

        python3 remove_supercategory.py

    Remove the supercategory using a specific dataset version:

        python3 remove_supercategory.py --version {DATASET_VERSION}

    where {DATASET_VERSION} is one of the following:
        - v1
        - v2
        - v3
"""

import argparse
import json
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Remove supercategories from different dataset versions.")
    parser.add_argument("--version", type=str, default="v3", choices=["v1", "v2", "v3"],
                        help="Dataset version (default: v3)")
    return parser.parse_args()


def load_annotations(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def remove_supercategory(data, unwanted_name):
    original_categories = data['categories']
    filtered_categories = [cat for cat in original_categories if cat['name'] != unwanted_name]

    if len(filtered_categories) == len(original_categories):
        print(f"[⚠️] No category named '{unwanted_name}' was removed. Double-check the name.")
    else:
        print(f"[✅] Removed '{unwanted_name}' category.")

    data['categories'] = filtered_categories
    return data


def save_annotations(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[✅] Cleaned file saved as: {output_file}")


def process_dataset(version):
    dataset_root = Path(__file__).resolve().parent.parent / f"datasets/supermarket_products_{version}"
    unwanted_name = "supermarket-product"

    for dir in ["train", "valid", "test"]:
        input_file = dataset_root / dir / "_annotations.coco.json"
        data = load_annotations(input_file)

        data = remove_supercategory(data, unwanted_name)

        output_file = dataset_root / dir / "annotations_without_supercategory.coco.json"
        save_annotations(data, output_file)


def main():
    args = parse_arguments()
    process_dataset(args.version)


if __name__ == "__main__":
    main()

