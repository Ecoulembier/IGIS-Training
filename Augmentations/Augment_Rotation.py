#!/usr/bin/env python3
"""
This script performs random rotation augmentation on images and their corresponding masks.

Usage: python Augment_Rotation.py /path/to/trainset

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
import sys
import cv2
import random
import numpy as np
from tqdm import tqdm

def get_rotation_angle():
    """Generate a random rotation angle between -45 and 45 degrees, excluding [-5,5]"""
    rotation_ranges = [(-45, -5), (5, 45)]
    chosen_range = random.choice(rotation_ranges)
    return random.uniform(*chosen_range)

def main():
    # Handle command line arguments
    if len(sys.argv) != 2:
        print("Error: Please provide the path to the dataset directory")
        print("Usage: python Augment_Rotation.py /path/to/trainset")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    
    # Configure paths
    input_img_dir = os.path.join(input_dir, 'img')
    input_mask_dir = os.path.join(input_dir, 'masks_machine')
    output_dir = os.path.join(os.path.dirname(input_dir), 'rotated_images')
    output_img_dir = os.path.join(output_dir, 'img')
    output_mask_dir = os.path.join(output_dir, 'masks_machine')
    
    # Validate directory structure
    if not os.path.exists(input_img_dir):
        print(f"Error: Missing 'img' directory in {input_dir}")
        sys.exit(1)
    if not os.path.exists(input_mask_dir):
        print(f"Error: Missing 'masks_machine' directory in {input_dir}")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Process images
    image_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith('.png')]
    print(f"Found {len(image_files)} images to process")
    
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Get paths (consistent with your other scripts)
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(input_img_dir, img_file)
            mask_path = os.path.join(input_mask_dir, img_file)
            
            # Load images
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None or mask is None:
                print(f"Skipping {img_file}: Could not load image or mask")
                continue
            if image.shape[:2] != mask.shape:
                print(f"Skipping {img_file}: Image and mask size mismatch")
                continue
            
            # Generate and apply rotation
            angle = get_rotation_angle()
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            
            rotated_img = cv2.warpAffine(
                image, M, (w, h), 
                borderMode=cv2.BORDER_REPLICATE
            )
            rotated_mask = cv2.warpAffine(
                mask, M, (w, h), 
                flags=cv2.INTER_NEAREST, 
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Save results
            new_name = f"{base_name}_rotated.png"
            cv2.imwrite(os.path.join(output_img_dir, new_name), rotated_img)
            cv2.imwrite(os.path.join(output_mask_dir, new_name), rotated_mask)
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    print(f"Processing complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
