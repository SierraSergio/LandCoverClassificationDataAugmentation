import os
import mmcv
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mmseg.apis import init_model, inference_model, show_result_pyplot

# Define the color palette
palette = [[0, 255, 0], [1, 128, 1], [255, 255, 255], [0, 0, 0], [255, 0, 0]]

def colorize_mask(mask, palette):
    """Colorize a mask according to a given palette."""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_mask[mask == label] = color
    return color_mask

# Initialize the model from the config and checkpoint
checkpoint_path = ''
cfg = './vis_data/config.py'
model = init_model(cfg, checkpoint_path, 'cuda:0')

# Define the paths for the input and output folders
input_folder = ''
output_folder_colorized = ''
output_folder_blacked = ''

# Ensure the output folder exists
os.makedirs(output_folder_colorized, exist_ok=True)
os.makedirs(output_folder_blacked, exist_ok=True)

# Iterate over all images in the input folder
for img_name in os.listdir(input_folder):
    # Construct the full path of the image
    img_path = os.path.join(input_folder, img_name)

    # Ensure the file is an image
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        try:
            # Read the image
            img = mmcv.imread(img_path)

            # Perform inference
            result = inference_model(model, img)

            # Extract the mask from the inference result
            mask = result.pred_sem_seg.data.cpu().numpy().astype(np.uint8).squeeze()

            # Define the output file path for the mask
            mask_output_path = os.path.join(output_folder_blacked, f"mask_{os.path.splitext(img_name)[0]}.tif")
            cv2.imwrite(mask_output_path, mask)

            # Colorize the mask using the defined palette
            color_mask = colorize_mask(mask, palette)

            # Define the output file path for the colorized mask
            mask_output_colorized_path = os.path.join(output_folder_colorized, f"mask_{os.path.splitext(img_name)[0]}.tif")

            # Save the colorized mask as an image using OpenCV
            cv2.imwrite(mask_output_colorized_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
