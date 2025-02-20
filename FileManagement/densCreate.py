import numpy as np
import cv2
import json
import os

def generate_density_map(points, radius, img_size):
    H, W = img_size
    density_map = np.zeros((H, W))

    for point in points:
        x, y = point
        window_size = int(radius * 2)
        gaussian_window = cv2.getGaussianKernel(window_size, radius / 4)
        gaussian_window = np.outer(gaussian_window, gaussian_window.T)

        x_start = max(0, x - radius)
        x_end = min(W, x + radius)
        y_start = max(0, y - radius)
        y_end = min(H, y + radius)

        density_map[y_start:y_end, x_start:x_end] += gaussian_window[
            :y_end - y_start, :x_end - x_start
        ]

    return density_map

def process_image(json_data, image_key, save_directory):
    image_data = json_data[image_key]
    points = image_data['points']
    radius = image_data['r'][0]  # assuming both r values are the same for simplicity
    img_path = image_data['img_path']
    img_size = (image_data['H'], image_data['W'])

    density_map = generate_density_map(points, radius, img_size)

    # Ensure save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Save the density map as a .npy file
    npy_path = os.path.join(save_directory, os.path.basename(img_path).replace('.jpg', '.npy'))
    np.save(npy_path, density_map)
    print(f"Density map saved at: {npy_path}")

def main(json_filepath, save_directory):
    with open(json_filepath, 'r') as f:
        json_data = json.load(f)

    for image_key in json_data.keys():
        process_image(json_data, image_key, save_directory)

# Replace 'your_json_file_path.json' with the actual path to your JSON file
# Replace 'your_save_directory' with the directory path where you want to save .npy files
main('./data-final/annotations.json', './data-final/density_map_adaptive_V1')
