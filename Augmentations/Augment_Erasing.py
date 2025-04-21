#!/usr/bin/env python3
"""
This script is used to augment images through random region erasing.

Usage: python Augment_Erasing.py /path/to/trainset

Directory structure must contain:
<trainset>
├── img
│   ├── img1.png
│   ├── img2.png
│   └── ...
└── masks_machine
    ├── img1.png
    ├── img2.png
    └── ...
"""

import os
import random
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

def random_erasing(image, mask, value=0):
    width, height = image.size
    erase_size = int(min(width, height) / 3)

    x = random.randint(0, width - erase_size)
    y = random.randint(0, height - erase_size)

    erase_area = (x, y, x + erase_size, y + erase_size)

    image_array = np.array(image)
    mask_array = np.array(mask)

    image_array[y:y+erase_size, x:x+erase_size] = value
    mask_array[y:y+erase_size, x:x+erase_size] = 0

    return Image.fromarray(image_array), Image.fromarray(mask_array)

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Error: Please provide the path to the dataset directory")
        print("Usage: python Augment_Erasing.py /path/to/trainset")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    
    # Configure paths
    input_img_dir = os.path.join(input_dir, 'img')
    input_mask_dir = os.path.join(input_dir, 'masks_machine')
    output_dir = os.path.join(os.path.dirname(input_dir), 'erased_images')
    
    # Validate input paths
    if not os.path.exists(input_img_dir):
        print(f"Error: Missing 'img' directory in {input_dir}")
        sys.exit(1)
    if not os.path.exists(input_mask_dir):
        print(f"Error: Missing 'masks_machine' directory in {input_dir}")
        sys.exit(1)
    
    # Create output directories
    output_img_dir = os.path.join(output_dir, 'img')
    output_mask_dir = os.path.join(output_dir, 'masks_machine')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Process images
    image_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith('.png')]
    print(f"Found {len(image_files)} images to process...")

    for img_file in tqdm(image_files, desc='Processing images'):
        try:
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(input_img_dir, img_file)
            mask_path = os.path.join(input_mask_dir, img_file)
            
            with Image.open(img_path) as img, Image.open(mask_path) as mask:
                if img.size != mask.size:
                    raise ValueError(f"Size mismatch between {img_file} and its mask")
                
                erased_img, erased_mask = random_erasing(img, mask)
                
                new_name = f"{base_name}_erased.png"
                erased_img.save(os.path.join(output_img_dir, new_name))
                erased_mask.save(os.path.join(output_mask_dir, new_name))
                
        except Exception as e:
            print(f"Skipping {img_file}: {str(e)}")

    print(f"Processing complete! Output saved to {output_dir}")

if __name__ == "__main__":
    main()
