#!/usr/bin/env python3
"""
This script is used to crop the images directly taken by the cameras on the IGIS device to obtain
images that contain the growth plate itself.
This script should be placed inside the folder that contains __ALL__ images to be processed.
The IGIS works with 10 plates and considers all places to be used.
10 subdirectories will be made called "0" to "9" for each plate.
To crop the images, there should be a file in the same folder called "Crop_configs.txt" which defines
the cropping box for each plate with the following format:
    <PlateNumber>,<BX>,<BY>,<Width>,<Height>
The coordinates are obtained by analysis of a box drawn around a plate in ImageJ, which ImageJ can
calculate under the tab Analyze-->Measure. This needs to be done once for each plate position, taking
into account that each plate on the same position appears approximately on the exact same spot.
"""

import os
from PIL import Image

# Get the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load crop coordinates from configuration file (converting ImageJ BX, BY, Width, Height coords to bottom_left, upper_left, upper_right, bottom_right)
crop_config_file = os.path.join(script_dir, "Crop_configs.txt")
crop_dict = {}

if os.path.exists(crop_config_file):
    with open(crop_config_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 5 and parts[0].isdigit():
                digit = parts[0]  # Last digit of filename
                BX, BY, width, height = map(int, parts[1:])  # Read ImageJ format
                
                # Convert to PIL format (left, upper, right, lower)
                crop_box = (BX, BY, BX + width, BY + height)
                crop_dict[digit] = crop_box
            else:
                print("Error: incorrect Crop_config.txt file format!\nUse \"<PlateNumber>,<TopLeftX>,<TopLeftY>,<Width>,<Height>\" per line!")
                exit(1)
else:
    print("Error: Crop_configs.txt not found!")
    exit(1)

for key, value in crop_dict.items():                                # Print out received crop dimensions
    print(f"Registered plate {key} with box dimensions {value}.")

# Ensure subfolders (0-9) exist, make if necessary
for i in range(10):
    os.makedirs(os.path.join(script_dir, str(i)), exist_ok=True)

# Count the total number of images to process
image_files = [f for f in os.listdir(script_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
total_images = len(image_files)
print(f"Total images to process: {total_images}")

# Crop each PNG and/or JPG image
processed_count = 0

for filename in image_files:
    if filename.lower().endswith((".png", ".jpg")):
        image_path = os.path.join(script_dir, filename)
        
        # Extract last digit of the filename, which defines the plate position
        name_without_ext = os.path.splitext(filename)[0]
        last_digit = name_without_ext[-1]

        if last_digit.isdigit() and last_digit in crop_dict:
            crop_box = crop_dict[last_digit]  # Get crop coordinates
            subfolder = os.path.join(script_dir, last_digit)  # Choose folder based on last digit
            
            # Open and crop image
            with Image.open(image_path) as img:
                cropped_img = img.crop(crop_box)

                # Define new filename with .png extension and save it
                new_filename = f"{name_without_ext}.png"
                output_path = os.path.join(subfolder, new_filename)
                cropped_img.save(output_path, format="PNG")
            
            # Update on cropping progress
            processed_count += 1
            print(f"Processed {processed_count}/{total_images}.")

        else:
            print(f"Skipping {filename}: No valid crop settings")

print("All images processed successfully!")
