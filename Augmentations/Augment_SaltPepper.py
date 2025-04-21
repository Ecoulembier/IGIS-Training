#!/usr/bin/env python3
"""
This script adds salt-and-pepper noise to images while preserving their masks.

Usage: python Augment_SaltPepper.py /path/to/trainset

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
import numpy as np
from PIL import Image
from tqdm import tqdm

def add_salt_and_pepper(image, amount=0.02):
    """Add salt-and-pepper noise to an image"""
    img_array = np.array(image)
    h, w, c = img_array.shape
    num_salt = np.ceil(amount * img_array.size * 0.5)
    num_pepper = np.ceil(amount * img_array.size * 0.5)
    
    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 255

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 0
    
    return Image.fromarray(img_array)

def main():
    # Handle command line arguments
    if len(sys.argv) != 2:
        print("Error: Please provide the path to the dataset directory")
        print("Usage: python Augment_SaltPepper.py /path/to/trainset")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    
    # Configure paths
    input_img_dir = os.path.join(input_dir, 'img')
    input_mask_dir = os.path.join(input_dir, 'masks_machine')
    output_dir = os.path.join(os.path.dirname(input_dir), 'salt_pepper_images')
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
    
    for img_file in tqdm(image_files, desc='Processing images'):
        try:
            # Get paths (consistent with your other scripts)
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(input_img_dir, img_file)
            mask_path = os.path.join(input_mask_dir, img_file)
            
            # Load and process image
            with Image.open(img_path) as img:
                augmented_img = add_salt_and_pepper(img)
            
            # Save augmented image
            new_name = f"{base_name}_SP.png"
            augmented_img.save(os.path.join(output_img_dir, new_name))
            
            # Copy corresponding mask
            if os.path.exists(mask_path):
                with Image.open(mask_path) as mask:
                    mask.save(os.path.join(output_mask_dir, new_name))
            else:
                print(f"Warning: Mask not found for {img_file}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    print(f"Processing complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
