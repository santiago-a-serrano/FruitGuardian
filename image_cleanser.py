import os
from PIL import Image

# Removes those images that are "cropped". WARNING: Deletes the files.
def check_transparent_edges(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        for x in range(width):
            if img.getpixel((x, 0))[3] != 0 or img.getpixel((x, height - 1))[3] != 0:
                return False
        for y in range(height):
            if img.getpixel((0, y))[3] != 0 or img.getpixel((width - 1, y))[3] != 0:
                return False
    return True

def find_and_delete_images_without_transparent_edges(directory):
    transparent_count = 0
    non_transparent_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".png"):
                image_path = os.path.join(root, file)
                if check_transparent_edges(image_path):
                    transparent_count += 1
                else:
                    non_transparent_count += 1
                    print(f"Deleting: {image_path}")
                    os.remove(image_path)
    print(f"Images with transparent edges: {transparent_count}")
    print(f"Images without transparent edges: {non_transparent_count}")

# Usage example
directory = "dataset/no_background"
find_and_delete_images_without_transparent_edges(directory)