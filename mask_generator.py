from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
import cv2
from skimage.measure import label

class MaskGenerator():
    # Other models: "vit_b" "sam_vit_b_01ec64.pth"
    #               "vit_h" "sam_vit_h_4b8939.pth"
    #               "vit_l" "sam_vit_l_0b3195.pth"
    def __init__(self, sam_model="vit_h", checkpoint_path="sam_vit_h_4b8939.pth"):
        sam = sam_model_registry[sam_model](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(sam)

    # image_path example: "test_baskets/image/0.png"
    # point_coords_path example: "test_baskets/fpos/0.np.npy"
    # output_folder example: "intermediate_data/masks"
    def generate_masks(self, image_path, point_coords_path, output_folder):
        predictor = self.predictor
        image = np.array(Image.open(image_path))
        predictor.set_image(image)

        point_coords = np.load(point_coords_path)

        object_masks = []

        # Generate a mask for each object separately
        for point in point_coords:
            point_label = np.array([1])  # Single point label for each object
            
            masks, scores, logits = predictor.predict(
                point_coords=np.array([point]),
                point_labels=point_label,
                multimask_output=False  # Set multimask_output to False
            )
            
            # Select the mask with the highest score
            best_mask = masks[np.argmax(scores)]
            
            object_masks.append(best_mask)

        # Visualize the individual object masks
        for i, obj_mask in enumerate(object_masks):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self._show_mask(obj_mask, plt.gca())
            np.save(f'{output_folder}/output_mask_{i+1}.npy', obj_mask)
            plt.title(f"Object {i+1}", fontsize=18)
            plt.axis('off')
            plt.savefig(f'{output_folder}/../visual_masks/mask_{i+1}.png', bbox_inches='tight', pad_inches=0)


    def _show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def _show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def _show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))