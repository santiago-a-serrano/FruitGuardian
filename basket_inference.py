from mask_generator import MaskGenerator
from fruit_extractor import extract_one_fruit
from fruit_classifier import get_model
import os
import torch
from PIL import Image

import numpy as np
from PIL import Image

def _apply_masks_and_save(image_path, rotten_mask_paths, fresh_mask_paths, output_path, copied_image_path):
    # Load the image
    image = Image.open(image_path).convert("L")
    image = image.convert("RGBA")
    image_array = np.array(image)

    # Save the original image
    Image.open(image_path).save(copied_image_path)

    # Create a red color mask with half opacity
    red_color = np.array([255, 0, 0, 128], dtype=np.uint8)
    green_color = np.array([0, 255, 0, 128], dtype=np.uint8)

    # Iterate over the mask paths
    for mask_path in rotten_mask_paths:
        # Load the boolean mask
        mask_array = np.load(mask_path)

        # Create a blank RGBA mask
        rgba_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)
        rgba_mask[mask_array > 0] = red_color

        # Alpha-blend the mask with the image
        mask_image = Image.fromarray(rgba_mask)
        image = Image.alpha_composite(image, mask_image)

    # # Iterate over the mask paths
    # for mask_path in fresh_mask_paths:
    #     # Load the boolean mask
    #     mask_array = np.load(mask_path)

    #     # Create a blank RGBA mask
    #     rgba_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)
    #     rgba_mask[mask_array > 0] = green_color

    #     # Alpha-blend the mask with the image
    #     mask_image = Image.fromarray(rgba_mask)
    #     image = Image.alpha_composite(image, mask_image)

    # Save the resulting image
    image.save(output_path)

# # Example usage
# image_path = "path/to/your/image.jpg"
# mask_paths = ["path/to/mask1.png", "path/to/mask2.png", "path/to/mask3.png"]
# output_path = "path/to/output/image.jpg"

# _apply_masks_and_save(image_path, mask_paths, output_path)

# example: basket_id = 50
def infer_one_basket(baskets_path="baskets/test_baskets", basket_id=50, sam_model="vit_h", checkpoint_path="sam_vit_h_4b8939.pth"):
    # generate the masks (segmentation) of each fruit
    mask_generator = MaskGenerator(sam_model, checkpoint_path)
    mask_generator.generate_masks(baskets_path + f"/image/{basket_id}.png", 
                                  baskets_path + f"/fpos/{basket_id}.np.npy",
                                  "intermediate_data/masks")

    # for all fruits in the basket, get each one of them individually
    folder_path = "intermediate_data/masks"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            extract_one_fruit(baskets_path + f"/image/{basket_id}.png", file_path, f"intermediate_data/separated_fruits/{filename}.png")

    # infer the class of each fruit, and save the rotten and fresh ones in arrays for later use
    class_labels = ["Fresh", "Rotten"]
    rotten_fruits = []
    fresh_fruits = []
    model = get_model()
    folder_path = "intermediate_data/separated_fruits"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with torch.no_grad():
                image = Image.open(file_path)
                # Such a small size might correspond to a wrongly-segmented element that is not a fruit
                if image.size[0] < 32 or image.size[1] < 32:
                    continue
                output = model(image)
                _, class_index = torch.max(output, 1)
                label = class_labels[class_index.item()]
                if label == "Rotten":
                    rotten_fruits.append("intermediate_data/masks/" + filename[0:-4])
                elif label == "Fresh":
                    fresh_fruits.append("intermediate_data/masks/" + filename[0:-4])

                # print(filename, label)

    # Final output
    _apply_masks_and_save(baskets_path + f"/image/{basket_id}.png", rotten_fruits, fresh_fruits, f"model_output.png", f"model_input.png")
    print(f"Rotten fruits in basket {basket_id}: {len(rotten_fruits)}")
    print(f"Fresh fruits in basket {basket_id}: {len(fresh_fruits)}")
    print(f"Percentage of rotten fruits in basket {basket_id}: {len(rotten_fruits) / (len(rotten_fruits) + len(fresh_fruits)) * 100:.2f}%")

    return len(fresh_fruits), len(rotten_fruits)


    
infer_one_basket()