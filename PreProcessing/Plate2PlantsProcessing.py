#!/usr/bin/env python3

import os
from PIL import Image

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the directory containing crop config files
crop_config_dir = os.path.join(script_dir, "Ref_crop_coords")

# Dictionary to hold crop coordinates for each subfolder
crop_dict = {str(i): {} for i in range(10)}  # Keys = "0" to "9"

# Load cropping regions from files
if os.path.exists(crop_config_dir):
    for filename in os.listdir(crop_config_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(crop_config_dir, filename)
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) == 6 and parts[0].isdigit():
                        subfolder, crop_id = parts[0], int(parts[1])
                        BX, BY, width, height = map(int, parts[2:])
                        
                        # Convert to PIL format (left, upper, right, lower) and store
                        crop_box = (BX, BY, BX + width, BY + height)
                        crop_dict[subfolder][crop_id] = crop_box
                    else:
                        print(f"Error: incorrect format in file {filename}!")
                        print("Use \"<PlateNumber>,<PlantPosition>,<TopLeftX>,<TopLeftY>,<Width>,<Height>\" per line!")
                        exit(1)
else:
    print("Error: Ref_crop_coords directory not found!")
    exit(1)

for key, value in crop_dict.items():
    print(f"Received {len(value)}/12 coordinates for folder {key}.")

# Process images in each subfolder (0-9)
for i in range(10):
    subfolder = os.path.join(script_dir, str(i))
    subfolder_key = str(i)  # Convert to string to match dictionary keys

    if os.path.exists(subfolder) and subfolder_key in crop_dict:
        image_files = [f for f in os.listdir(subfolder) if f.lower().endswith(".png")]
        total_images = len(image_files)
        print(f"Total of {total_images} images to process in folder {i+1}...")
        processed_images = 0

        for filename in image_files:
            file_path = os.path.join(subfolder, filename)

            with Image.open(file_path) as img:
                for crop_id, crop_box in crop_dict[subfolder_key].items():
                    cropped_img = img.crop(crop_box)

                    # Construct output filename ("image_plant1.png")
                    name_without_ext = os.path.splitext(filename)[0]
                    output_filename = f"{name_without_ext}_plant{crop_id}.png"
                    output_path = os.path.join(subfolder, output_filename)
                    cropped_img.save(output_path)

            # Update on cropping process
            processed_images += 1
            print(f"Processed {processed_images}/{total_images} in folder {i+1}/10...")
    
    print(f"Processed folder {i+1}/10!")

print("All images processed successfully!")
