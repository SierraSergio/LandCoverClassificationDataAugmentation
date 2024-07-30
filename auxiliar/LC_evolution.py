import os
import rasterio
from collections import defaultdict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

    # Count the pixels in each class
    unique, counts = np.unique(mask_array, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    return pixel_counts

def count_pixels_shrub(image_path):
    # Open the mask image
    mask = Image.open(image_path)
    # Convert the image to a numpy array
    mask_array = np.array(mask)

    # Count the pixels in each class
    unique, counts = np.unique(mask_array, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    # Calculate the percentage of pixels for class 0 and class 1 if they exist
    total_pixel_count = sum(pixel_counts.values())
    counts = {
        0: 0.0,
        1: 0.0
    }

    if 0 in pixel_counts:
        counts[0] = round(pixel_counts[0])

    if 1 in pixel_counts:
        counts[1] = round(pixel_counts[1])

    return counts

def compare_images(bounds_dict, mask_path, classification, output_dir):
    evolution_data = defaultdict(lambda: defaultdict(list))

    # Specific colors for each class
    class_colors = {
        0: 'darkgreen',
        1: 'lightgreen',
        2: 'black',
        3: 'gray',
        4: 'red'
    }

    # Display images that have the same bounds
    for bounds, filenames in bounds_dict.items():
        if len(filenames) > 1:
            print(f"The following images have the same bounds {bounds},{filenames[0][-9:-4]}:")
            for filename in filenames:
                if classification == "model":
                    img_path = f'{mask_path}/mask_{filename}'
                else:
                    img_path = f'{mask_path}/{filename}'
                pixel_counts = count_pixels_by_class(img_path)
                total_pixel_count = sum(pixel_counts.values())
                percentages = {k: round(v * 100 / total_pixel_count, 2) for k, v in pixel_counts.items()}
                date = filename.split('_')[1][:8]
                print(date, percentages)
                pixel_percentages = count_pixels_shrub(img_path)
                print(pixel_percentages)

                # Store evolution data
                for class_id, percentage in percentages.items():
                    evolution_data[bounds][class_id].append((date, percentage))

    # Generate evolution graphs
    for bounds, class_data in evolution_data.items():
        plt.figure(figsize=(12, 6))
        for class_id, data in class_data.items():

            classes = {0: 'Shrub', 1: 'Grass', 2: 'Others', 3: 'Shadows', 4: 'Burned'}
            dates, percentages = zip(*sorted(data))  # Sort by date
            plt.plot(dates, percentages, label=f'{classes[class_id]}', color=class_colors[class_id])

            # Annotate values at peaks
            max_value = max(percentages)
            min_value = min(percentages)
            max_date = dates[percentages.index(max_value)]
            min_date = dates[percentages.index(min_value)]
            plt.annotate(f'{max_value}%', xy=(max_date, max_value), xytext=(max_date, max_value + 5),
                         arrowprops=dict(facecolor='green', shrink=0.05))
            plt.annotate(f'{min_value}%', xy=(min_date, min_value), xytext=(min_date, min_value - 5),
                         arrowprops=dict(facecolor='red', shrink=0.05))

        plt.ylim(0, 100)  # Set upper limit of y-axis to 100%
        if classification == "model":
            plt.title(f'Model classification evolution in coordinates {bounds}', fontsize=18)
        else:
            plt.title(f'{classification} classification evolution in coordinates {bounds}', fontsize=18)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Percentage of pixels', fontsize=14)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.show()

        # Save the plot to a file
        output_filename = f'evolution_bounds_{bounds}.png'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        plt.close()

# Call the function with the appropriate image directory
classification = "model"  # "Laura"

if classification == "model":
    images_path = ''
    mask_path = ''
    output_dir = ''
else:
    images_path = ''
    mask_path = ''
    output_dir = ''

bounds_dict = process_tiff_images(images_path)
compare_images(bounds_dict, mask_path, classification, output_dir)
