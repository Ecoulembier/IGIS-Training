#!/usr/bin/env python3
"""
This script is used to augment the original images through a random cropping step. The random crop always has
half the size of the original image, with the cropping window placed randomly on the original image.

Usage: python Augment_Cropping.py /path/to/trainset

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
"""

import os
import random
import sys
from PIL import Image
from tqdm import tqdm

def main():
    # Configuration
    CROP_RATIO = 0.5  # Half of original size
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python Augment_Cropping.py /path/to/trainset")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found - {input_dir}")
        sys.exit(1)
    
    # Set up paths
    INPUT_IMAGE_DIR = os.path.join(input_dir, 'img')
    INPUT_MASK_DIR = os.path.join(input_dir, 'masks_machine')
    OUTPUT_DIR = os.path.join(os.path.dirname(input_dir), 'cropped_images')
    
    # Verify input directories exist
    if not os.path.isdir(INPUT_IMAGE_DIR):
        print(f"Error: 'img' subdirectory not found in {input_dir}")
        sys.exit(1)
    if not os.path.isdir(INPUT_MASK_DIR):
        print(f"Error: 'masks_machine' subdirectory not found in {input_dir}")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_DIR, 'img'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'masks_machine'), exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith('.png')]
    print(f"Found {len(image_files)} images to process...")
    
    for img_file in tqdm(image_files, desc='Processing images'):
        img_path = os.path.join(INPUT_IMAGE_DIR, img_file)
        mask_path = os.path.join(INPUT_MASK_DIR, f"{os.path.splitext(img_file)[0]}_mask.png")
        
        try:
            with Image.open(img_path) as img, Image.open(mask_path) as mask:
                # Verify dimensions match
                if img.size != mask.size:
                    raise ValueError(f"Image and mask size mismatch: {img.size} vs {mask.size}")
    
                width, height = img.size
                crop_width = int(width * CROP_RATIO)
                crop_height = int(height * CROP_RATIO)
    
                # Calculate maximum starting positions to ensure:
                # 1. Crop stays within image bounds
                # 2. Origin remains in upper-left quadrant
                max_x = min(width - crop_width, width // 2)
                max_y = min(height - crop_height, height // 2)
                
                if max_x < 0 or max_y < 0:
                    raise ValueError(f"Image {img_file} too small for {CROP_RATIO} ratio crop")
    
                # Generate random origin in upper-left quadrant
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
    
                # Calculate crop boundaries
                crop_box = (
                    x,  # Left (from quadrant 1)
                    y,  # Top (from quadrant 1)
                    x + crop_width,  # Right (extending into quadrant 4)
                    y + crop_height  # Bottom (extending into quadrant 4)
                )
    
                # Perform cropping
                cropped_img = img.crop(crop_box)
                cropped_mask = mask.crop(crop_box)
    
                # Save results
                base_name = os.path.splitext(img_file)[0]
                new_base = f"{base_name}_cropped.png"
                cropped_img.save(os.path.join(OUTPUT_DIR, 'img', new_base))
                cropped_mask.save(os.path.join(OUTPUT_DIR, 'masks_machine', new_base))
    
        except Exception as e:
            print(f"Skipping {img_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
