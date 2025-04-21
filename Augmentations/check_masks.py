#!/usr/bin/env python3
"""
This script visualizes the binary class by taking a mask file as input.
The mask file should only have 2 classes with RGB pixels values of (0, 0, 0) for the background
and (1, 1, 1) for the object of interest. Running this script returns an image with a visualized
red mask against a black background.
This script should be placed inside the folder that contains the masks.
"""
from PIL import Image
from pathlib import Path
import os

# Set the working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load mask
mask = Image.open("image_name.png").convert("L")

# Convert mask to RGB
mask_colored = mask.convert("RGB")

# Change white pixels to red (or any color)
mask_data = mask_colored.load()
for y in range(mask_colored.height):
    for x in range(mask_colored.width):
        if mask_data[x, y] == (1, 1, 1):  # If value is 1
            mask_data[x, y] = (255, 0, 0)  # Change to red

# Show mask
mask_colored.show()
