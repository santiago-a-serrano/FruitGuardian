import os
import random
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
from tqdm import tqdm

# they correspond to the folders in which the images are classified
class_labels = {"freshapples": 1, "freshoranges": 2, "rottenapples": 3, "rottenoranges": 4}

def generate_class_probabilities():
    # Define the range of probabilities for fresh fruits
    fresh_probability_ranges = {
        "freshapples": (0.35, 0.5),
        "freshoranges": (0.35, 0.5)
    }

    # Generate random probabilities for fresh fruits
    class_probabilities = {}
    remaining_probability = 1.0
    for folder, probability_range in fresh_probability_ranges.items():
        probability = random.uniform(probability_range[0], min(probability_range[1], remaining_probability))
        class_probabilities[folder] = probability
        remaining_probability -= probability

    # Calculate the total probability for rotten fruits
    rotten_probability = remaining_probability

    # Randomly divide the rotten probability between rotten apples and rotten oranges
    rottenapples_probability = random.uniform(0, rotten_probability)
    rottenoranges_probability = rotten_probability - rottenapples_probability

    # Assign the probabilities to the rotten fruit classes
    class_probabilities["rottenapples"] = rottenapples_probability
    class_probabilities["rottenoranges"] = rottenoranges_probability

    return class_probabilities

def make_one_basket(fruit_dir, baskets_dir, output_dir, basket_name, mask_name, metadata_name, pos_name, num_images):
    # Get list of fruit images for each folder
    fruit_images_by_folder = {}
    for root, dirs, files in os.walk(fruit_dir):
        if os.path.basename(root) not in ["freshbanana", "rottenbanana"]:
            folder = os.path.basename(root)
            fruit_images_by_folder[folder] = [os.path.join(root, file) for file in files if file.endswith(".png")]

    # To store the metadata in a txt file:
    class_counts = {"freshapples": 0, 
                    "freshoranges": 0, 
                    "rottenapples": 0,
                    "rottenoranges": 0}

    # Get list of basket images
    basket_images = [os.path.join(baskets_dir, file) for file in os.listdir(baskets_dir) if file.endswith(".png")]

    # Randomly select a basket image
    basket_image_path = random.choice(basket_images)
    basket_image = Image.open(basket_image_path)

    # Get the original basket dimensions
    basket_width, basket_height = basket_image.size

    # Create a mask image with the same size as the basket image
    mask_image = Image.new("RGB", (basket_width, basket_height), (0, 0, 0))

    # Calculate the padding
    vertical_padding = int(basket_height * 0.15)
    horizontal_padding = int(basket_width * 0.15)

    # Set the grid size
    grid_width = 5
    grid_height = 7

    fruit_width = (basket_width - 2 * horizontal_padding) // grid_width
    fruit_height = (basket_height - 2 * vertical_padding) // grid_height

    fruit_size = min(fruit_width, fruit_height)

    # Calculate the space between fruits
    fruit_spacing_x = (basket_width - 2 * horizontal_padding - grid_width * fruit_size) // (grid_width - 1)
    fruit_spacing_y = (basket_height - 2 * vertical_padding - grid_height * fruit_size) // (grid_height - 1)

    # Create a list of grid positions
    grid_positions = [(i, j) for i in range(grid_height) for j in range(grid_width)]
    random.shuffle(grid_positions)

    # Define the probability distribution for each class
    class_probabilities = generate_class_probabilities()

    # Create a list of fruit data with the desired probabilities
    fruit_data = []
    for folder, images in fruit_images_by_folder.items():
        if folder in class_probabilities:
            num_samples = int(class_probabilities[folder] * num_images)
            fruit_data.extend([(image, folder) for image in random.sample(images, num_samples)])
            class_counts[folder] += num_samples

    fruit_positions = []
    # Place fruit images on the grid in a random order
    for pos, (fruit_path, folder) in zip(grid_positions[:num_images], fruit_data[:num_images]):
        i, j = pos
        fruit_image = Image.open(fruit_path)
        # Randomly resize the fruit image
        size_multiplier = random.uniform(1.4, 1.6)
        fruit_aspect_ratio = fruit_image.width / fruit_image.height
        fruit_width = int(fruit_size * fruit_aspect_ratio * size_multiplier)
        fruit_height = int(fruit_size * size_multiplier)
        fruit_image = fruit_image.resize((fruit_width, fruit_height))
        # Randomly rotate the fruit image
        angle = random.randint(0, 359)
        fruit_image = fruit_image.rotate(angle, expand=True)

        x = horizontal_padding + j * (fruit_size + fruit_spacing_x) + (fruit_size - fruit_width) // 2 + random.randint(-15, 15)
        y = vertical_padding + i * (fruit_size + fruit_spacing_y) + (fruit_size - fruit_height) // 2 + random.randint(-15, 15)

        # FOR THE POINTS
        x_middle = x + fruit_width // 2
        y_middle = y + fruit_height // 2
        fruit_positions.append((x_middle, y_middle))

        # Paste the fruit mask onto the mask image with the class label
        class_label_color = (class_labels[folder], 0, 0)
        fruit_mask = Image.new("RGB", fruit_image.size, class_label_color)
        # Convert to NumPy array
        alpha = np.array(fruit_image.split()[-1])
        alpha[alpha >= 128] = 255
        alpha[alpha < 128] = 0
        alpha = Image.fromarray(alpha)
        fruit_mask.putalpha(alpha)
        np.set_printoptions(threshold=np.inf)
        mask_image.paste(fruit_mask, (x, y), mask=fruit_mask)
        # Paste the fruit image onto the basket image
        basket_image.paste(fruit_image, (x, y), fruit_image)

    # Save the final basket with fruits image
    basket_output_path = os.path.join(output_dir, basket_name)
    basket_image.save(basket_output_path)

    # Save the mask image
    mask_output_path = os.path.join(output_dir, mask_name)
    mask_image.save(mask_output_path)

    # Save the fruit coordinates
    fruit_positions_output_path = os.path.join(output_dir, pos_name)
    np.save(fruit_positions_output_path, fruit_positions)

    
    # Save the metadata
    metadata_output_path = os.path.join(output_dir, metadata_name)
    with open(metadata_output_path, 'w') as f:
        f.write(f"{basket_image_path}\n")
        for folder, count in class_counts.items():
            f.write(f"{folder}: {count}\n")



# 1000 test baskets
for i in tqdm(range(1000), desc="Generating test baskets"):
    make_one_basket("dataset/no_background/test", "empty_baskets", "NewFruitGuardian/baskets/test_baskets", f"image/{i}.png", f"only_for_deeplab/{i}.png", f"meta/{i}.txt", f"fpos/{i}.np", 35)

# # 1000 validation baskets
# for i in tqdm(range(1000), desc="Generating validation baskets"):
#     make_one_basket("dataset/no_background/train", "empty_baskets", "NewFruitGuardian/val_baskets", f"image/{i}.png", f"only_for_deeplab/{i}.png", f"meta/{i}.txt", f"fpos/{i}.np", 35)

# # 4000 train baskets
# for i in tqdm(range(4000), desc="Generating train baskets"):
#     make_one_basket("dataset/no_background/train", "empty_baskets", "NewFruitGuardian/train_baskets", f"image/{i}.png", f"only_for_deeplab/{i}.png", f"meta/{i}.txt", f"fpos/{i}.np", 35)