#!/usr/bin/env python3

"""
This script pads all images to the same size by adding a border of black pixels, so that the original image stays
centered in the padded image.
Before running this script, make sure that the folder containing the original, annotated images ONLY contains the
images that you want to augment, as the script will iterate over ALL .png images in the folder.

Usage: python image_padding.py /path/to/trainset

The directory structure must contain:
<trainset>
├── img
│   ├── img1.png
│   ├── img2.png
│   └── ...
└── masks_machine
    ├── img1_mask.png
    ├── img2_mask.png
    └── ...
The output consists of a folder called "padded", found within the <trainset> folder.
"""

import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import argparse
import sys

def find_max_dimensions(image_dir):
    max_width, max_height = 0, 0
    for filename in os.listdir(image_dir):
        if filename.lower().endswith('.png'):
            with Image.open(os.path.join(image_dir, filename)) as img:
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)
    return max_width, max_height

def pad_image(image, target_size):
    width, height = image.size
    delta_w = target_size[0] - width
    delta_h = target_size[1] - height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    # Ensure the image is RGB before padding
    if image.mode != "RGB":
        image = image.convert("RGB")
    padded_image = ImageOps.expand(image, padding, fill=(0, 0, 0))
    return padded_image

def pad_mask(mask, target_size):
    width, height = mask.size
    delta_w = target_size[0] - width
    delta_h = target_size[1] - height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    padded_mask = ImageOps.expand(mask, padding, fill=0)
    return padded_mask

def pad_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir):
    max_width, max_height = find_max_dimensions(image_dir)
    target_size = (max_width, max_height)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    for filename in tqdm(os.listdir(image_dir), desc="Padding images and masks"):
        if filename.lower().endswith('.png'):
            # Pad image
            image_path = os.path.join(image_dir, filename)
            try:
                with Image.open(image_path) as img:
                    padded_img = pad_image(img, target_size)
                    padded_img.save(os.path.join(output_image_dir, filename))
            except FileNotFoundError:
                print(f"Error: Image file not found: {image_path}")
                continue

            # Pad mask
            mask_path = os.path.join(mask_dir, filename)
            try:
                with Image.open(mask_path) as mask:
                    padded_mask = pad_mask(mask, target_size)
                    padded_mask.save(os.path.join(output_mask_dir, filename))
            except FileNotFoundError:
                print(f"Error: Mask file not found: {mask_path}")
                continue

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Pad images and masks in a dataset.")
    parser.add_argument("dataset_dir", help="Path to the root directory of the test set (containing img and masks_machine subdirectories)")
    args = parser.parse_args()

    # Construct paths from the dataset directory
    dataset_dir = args.dataset_dir
    image_dir = os.path.join(dataset_dir, 'img')
    mask_dir = os.path.join(dataset_dir, 'masks_machine')
    output_image_dir = os.path.join(dataset_dir, 'padded', 'img')
    output_mask_dir = os.path.join(dataset_dir, 'padded', 'masks_machine')

    # Check if the dataset directory exists
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    # Check if image and mask directories exist
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)
    if not os.path.isdir(mask_dir):
        print(f"Error: Mask directory not found: {mask_dir}")
        sys.exit(1)

    # Run the padding process
    pad_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir)
    print("Padding complete!")

if __name__ == "__main__":
    main()
