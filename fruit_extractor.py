from PIL import Image
import numpy as np

# original_img_path example: "test_baskets/image/0.png"
# fruit_mask_path example: "a/b/output_mask_0.npy"
# output_path example: "a/b/output_0.png"
def extract_one_fruit(original_img_path, fruit_mask_path, output_path):
    original_img = Image.open(original_img_path)
    mask_array = np.load(fruit_mask_path)
    masked_img = Image.fromarray(np.array(original_img) * mask_array[:, :, np.newaxis])

    non_empty_columns = np.where(mask_array.max(axis=0))[0]
    non_empty_rows = np.where(mask_array.max(axis=1))[0]
    cropped_box = (min(non_empty_columns), min(non_empty_rows), max(non_empty_columns), max(non_empty_rows))

    cropped_img = masked_img.crop(cropped_box)
    cropped_img.save(output_path)