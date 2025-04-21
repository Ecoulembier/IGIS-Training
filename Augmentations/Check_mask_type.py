from PIL import Image
import numpy as np
import sys

def analyze_mask(mask_path):
    """
    Analyzes a mask image to determine its format and pixel values.

    Args:
        mask_path (str): The path to the mask image.
    """
    try:
        mask = Image.open(mask_path).convert('L') #Open in greyscale
    except FileNotFoundError:
        print(f"Error: The file '{mask_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Could not open or process the image. {e}")
        sys.exit(1)

    print(f"Mask mode: {mask.mode}")

    # Get pixel values from the entire image
    pixel_values = list(mask.getdata())

    # Create a dictionary to store the count of each pixel value
    pixel_counts = {}
    for pixel_value in pixel_values:
        if pixel_value in pixel_counts:
            pixel_counts[pixel_value] += 1
        else:
            pixel_counts[pixel_value] = 1

    print("Pixel Value Counts:")
    for value, count in sorted(pixel_counts.items()):
        print(f"Value: {value}, Count: {count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <mask_path>")
        sys.exit(1)

    mask_path = sys.argv[1]
    analyze_mask(mask_path)
