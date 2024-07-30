'''
This script computes changes between images from different dates. Changes are generated as numeric values:
if there is a change from category 0 to 1, it is marked in the image as a 1; if the change is from 1 to 4, it will be marked as 14.
For common pixels, the chosen value is 255.
'''

import os
import rasterio
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2

def get_georeferencing_info(tiff_path, mask_path, classification):
    if classification == "model":
        image_name = f'mask_{os.path.basename(tiff_path)}'
        data = cv2.imread(os.path.join(mask_path, image_name))
    else:
        data = cv2.imread(os.path.join(mask_path, os.path.basename(tiff_path)))

    with rasterio.open(tiff_path) as dataset:
        bounds = dataset.bounds
        transform = dataset.transform
        crs = dataset.crs
        info = {
            'filename': os.path.basename(tiff_path),
            'bounds': bounds,
            'transform': transform,
            'crs': crs,
            'data': data
        }
        return info

def process_tiff_images(directory, mask_path, classification):
    georef_info_list = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(('tif', 'tiff')):
            tiff_path = os.path.join(directory, filename)
            info = get_georeferencing_info(tiff_path, mask_path, classification)
            georef_info_list.append(info)

    # Create a dictionary to group images by bounds
    bounds_dict = defaultdict(list)

    for info in georef_info_list:
        bounds = info['bounds']
        bounds_int = [int(bounds.left), int(bounds.bottom), int(bounds.right), int(bounds.top)]
        bounds_dict[tuple(bounds_int)].append(info)

    return bounds_dict

def compare_images(images_info):
    reference_image = images_info[0]['data']
    current_image = images_info[-1]['data']

    result_image = np.zeros_like(reference_image, dtype=object)

    for i in range(reference_image.shape[0]):
        for j in range(reference_image.shape[1]):
            result_image[i, j] = f"{reference_image[i, j][0]}{current_image[i, j][0]}" if reference_image[i, j].all() != \
                                                                                          current_image[
                                                                                              i, j].all() else str(
                255)
    print(result_image)
    return result_image

def save_georef(ref_image_path, target_image_path, output_image_path ):
    # Read the reference image
    with rasterio.open(ref_image_path) as ref_src:
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs

    # Read the target image
    with rasterio.open(target_image_path) as target_src:
        target_data = target_src.read()
        target_meta = target_src.meta

    # Update the target image's metadata with the georeferencing of the reference image
    target_meta.update({
        'transform': ref_transform,
        'crs': ref_crs
    })

    # Write the georeferenced image
    with rasterio.open(output_image_path, 'w', **target_meta) as dest:
        dest.write(target_data)

    print("Georeferenced image saved to", output_image_path)

def create_legend(result_image):
    unique_values = np.unique(result_image)

    legend = {
        '1': '',
        '2': '',
        '3': '',
        '4': '',
        '10': '',
        '12': '',
        '13': '',
        '14': '',
        '20': '',
        '21': '',
        '23': '',
        '24': '',
        '30': '',
        '31': '',
        '32': '',
        '34': '',
        '40': '',
        '41': '',
        '42': '',
        '43': '',
    }

    filtered_legend = {k: v for k, v in legend.items() if k in unique_values}

    return filtered_legend

def save_comparison_image(result_image, output_path):
    result_image = result_image.astype(np.uint8)
    Image.fromarray(result_image).save(output_path)


# Call the function with the appropriate image directory
classification = "model"  #

if classification == "model":
    images_path = ''
    mask_path = ''
    output_dir = ''
else:
    images_path = ''
    mask_path = ''
    output_dir = ''

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bounds_dict = process_tiff_images(images_path, mask_path, classification)

for bounds, images_info in bounds_dict.items():
    if len(images_info) > 1:
        result_image = compare_images(images_info)
        filename = images_info[0]['filename']
        output_path = os.path.join(output_dir, f'comparison_{filename}')
        save_comparison_image(result_image, output_path)
        ref_image_path = f'{images_path}/{filename}'
        save_georef(ref_image_path, output_path, output_path)
