#!/usr/bin/env python3

from PIL import Image
from pathlib import Path
import os

# Set the working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load mask
mask = Image.open("2024_07_28_2829_plant8.png").convert("L")

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
