import os
import rasterio
from collections import defaultdict
from PIL import Image
import numpy as np

def get_georeferencing_info(tiff_path):
    with rasterio.open(tiff_path) as dataset:
        bounds = dataset.bounds
        info = {
            'filename': os.path.basename(tiff_path),
            'bounds': bounds
        }
        return info

def process_tiff_images(directory):
    georef_info_list = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(('tif', 'tiff')):
            tiff_path = os.path.join(directory, filename)
            info = get_georeferencing_info(tiff_path)
            georef_info_list.append(info)

    # Create a dictionary to group images by bounds
    bounds_dict = defaultdict(list)

    for info in georef_info_list:
        bounds = info['bounds']
        bounds_int = [int(bounds.left), int(bounds.bottom), int(bounds.right), int(bounds.top)]
        bounds_dict[tuple(bounds_int)].append(info['filename'])

    return bounds_dict

def count_pixels_by_class(image_path):
    # Open the mask image
    mask = Image.open(image_path)
    # Convert the image to a numpy array
    mask_array = np.array(mask)

    # Count the pixels of each class
    unique, counts = np.unique(mask_array, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    return pixel_counts

def count_pixels_shrub(image_path):
    # Open the mask image
    mask = Image.open(image_path)
    # Convert the image to a numpy array
    mask_array = np.array(mask)

    # Count the pixels of each class
    unique, counts = np.unique(mask_array, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    # Calculate the number of pixels for class 0 and class 1 if they exist
    total_pixel_count = sum(pixel_counts.values())
    pixel_sums = {
        0: 0.0,
        1: 0.0
    }

    if 0 in pixel_counts:
        pixel_sums[0] = round(pixel_counts[0])

    if 1 in pixel_counts:
        pixel_sums[1] = round(pixel_counts[1])

    return pixel_sums

def compare_images(bounds_dict, mask_path):
    # Display images that have the same bounds
    for bounds, filenames in bounds_dict.items():
        if len(filenames) > 1:
            print(f"The following images have the same bounds {bounds},{filenames[0][-9:-4]}:")
            for filename in filenames:
                img_path = f'{mask_path}/mask_{filename}'
                pixel_counts = count_pixels_by_class(img_path)
                total_pixel_count = sum(pixel_counts.values())
                percentages = {k: round(v * 100 / total_pixel_count, 2) for k, v in pixel_counts.items()}
                print(filename.split('_')[1][:8], percentages)
                pixel_percentages = count_pixels_shrub(img_path)
                print(pixel_percentages)

# Call the function with the directory of the images
images_path = ''
mask_path = ''
bounds_dict = process_tiff_images(images_path)
compare_images(bounds_dict, mask_path)


