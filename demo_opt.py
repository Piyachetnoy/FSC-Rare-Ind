"""
Demo file for Few Shot Counting with Dot-based Density Visualization using MPS

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Modified: [Your Name]
Created: 19-Apr-2021
Last modified: [Date]
"""

import cv2
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import select_exemplar_rois
from PIL import Image
import os
import torch
import argparse
import torch.optim as optim
from utils import MincountLoss, PerturbationLoss
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# Visualization function with dot-based density
def visualize_output_with_accuracy_levels(image, heatmap, boxes, rslt_file, thresholds=(0.7, 0.65, 0.4), min_distance=10):
    """
    Visualize output with dots categorized into high, medium, and low density levels.
    
    Args:
        image: Original image (PIL format or tensor).
        heatmap: Output heatmap (tensor).
        boxes: Bounding boxes (tensor).
        rslt_file: Path to save the result.
        thresholds: Tuple of thresholds (high, medium, low) for dot categorization.
    """
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).cpu().numpy()

    heatmap = heatmap.squeeze().cpu().numpy()
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Extract dots with different density thresholds
    high_density_coords = peak_local_max(heatmap, min_distance, threshold_abs=thresholds[0])
    medium_density_coords = peak_local_max(heatmap, min_distance, threshold_abs=thresholds[1])
    low_density_coords = peak_local_max(heatmap, min_distance, threshold_abs=thresholds[2])

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Plot dots with different colors for accuracy levels
    if low_density_coords.size > 0:
        plt.scatter(
            low_density_coords[:, 1],
            low_density_coords[:, 0],
            color='red',
            s=15,
            label='Low Density',
            alpha=0.8,
        )
    if medium_density_coords.size > 0:
        plt.scatter(
            medium_density_coords[:, 1],
            medium_density_coords[:, 0],
            color='blue',
            s=15,
            label='Medium Density',
            alpha=0.8,
        )
    if high_density_coords.size > 0:
        plt.scatter(
            high_density_coords[:, 1],
            high_density_coords[:, 0],
            color='green',
            s=15,
            label='High Density',
            alpha=0.8,
        )
    
    # Draw bounding boxes
    for box in boxes:
        if box.numel() == 4:  # Check if the box has exactly 4 elements
            y1, x1, y2, x2 = box.int().cpu().numpy()
            plt.gca().add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='blue', facecolor='none', lw=2)
            )
        else:
            print("Skipping invalid box:", box)

    plt.title("Dot-based Density Visualization")
    plt.legend()
    plt.axis('off')
    plt.savefig(rslt_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"===> Visualized output with dots is saved to {rslt_file}")

def count_dots_with_adjusted_area(image, heatmap, thresholds=(0.7, 0.65, 0.4), min_distance=10):
    """
    Count the number of dots in high, medium, and low density categories, with an adjustable minimum distance.

    Args:
        image: Original image (PIL format or tensor).
        heatmap: Output heatmap (tensor).
        thresholds: Tuple of thresholds (high, medium, low) for dot categorization.
        min_distance: Minimum distance between detected peaks to count as separate dots.

    Returns:
        Counts of dots in high, medium, and low density categories.
    """
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).cpu().numpy()

    heatmap = heatmap.squeeze().cpu().numpy()
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Extract dots with different density thresholds and adjustable area
    high_density_coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=thresholds[0])
    medium_density_coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=thresholds[1])
    low_density_coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=thresholds[2])

    all_coords = np.vstack([high_density_coords, medium_density_coords, low_density_coords])
    unique_coords = np.unique(all_coords, axis=0)

    # Count total unique dots
    total_unique_dots = unique_coords.shape[0]
    print(f"Total Unique Dots: {total_unique_dots}")

    # Count dots in each category
    high_density_count = high_density_coords.shape[0]
    medium_density_count = medium_density_coords.shape[0]
    low_density_count = low_density_coords.shape[0]

    # Print the counts
    print("Dot Counts by Density Category (with adjusted area):")
    print(f"  High Density: {high_density_count} dots")
    print(f"  Medium Density: {medium_density_count} dots")
    print(f"  Low Density: {low_density_count} dots")

    return high_density_count, medium_density_count, low_density_count, total_unique_dots

parser = argparse.ArgumentParser(description="Few Shot Counting Demo code")
parser.add_argument("-i", "--input-image", type=str, required=True, help="/Path/to/input/image/file/")
parser.add_argument("-b", "--bbox-file", type=str, help="/Path/to/file/of/bounding/boxes")
parser.add_argument("-o", "--output-dir", type=str, default=".", help="/Path/to/output/image/file")
parser.add_argument("-m",  "--model_path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth", help="path to trained model")
parser.add_argument("-a",  "--adapt", action='store_true', help="If specified, perform test time adaptation")
parser.add_argument("-gs", "--gradient_steps", type=int,default=100, help="number of gradient steps for the adaptation")
parser.add_argument("-lr", "--learning_rate", type=float,default=1e-7, help="learning rate for adaptation")
parser.add_argument("-wm", "--weight_mincount", type=float,default=1e-9, help="weight multiplier for Mincount Loss")
parser.add_argument("-wp", "--weight_perturbation", type=float,default=1e-4, help="weight multiplier for Perturbation Loss")
parser.add_argument("-th", "--threshold", type=float, default=0.5, help="Density threshold for dot detection")

args = parser.parse_args()

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("0")
    print("===> Using MPS mode.")
else:
    device = torch.device("cpu")
    print("===> Using CPU mode.")

# Model setup
resnet50_conv = Resnet50FPN().to(device)
regressor = CountRegressor(6, pool='mean').to(device)
regressor.load_state_dict(torch.load(args.model_path, map_location=device))

resnet50_conv.eval()
regressor.eval()

# Input image and bounding boxes
image_name = os.path.basename(args.input_image)
image_name = os.path.splitext(image_name)[0]

if args.bbox_file is None:
    out_bbox_file = "{}/{}_box.txt".format(args.output_dir, image_name)
    fout = open(out_bbox_file, "w")

    im = cv2.imread(args.input_image)
    cv2.imshow('image', im)
    rects = select_exemplar_rois(im)

    rects1 = []
    for rect in rects:
        y1, x1, y2, x2 = rect
        rects1.append([y1, x1, y2, x2])
        fout.write("{} {} {} {}\n".format(y1, x1, y2, x2))

    fout.close()
    cv2.destroyWindow("Image")
    print("Selected bounding boxes are saved to {}".format(out_bbox_file))
else:
    with open(args.bbox_file, "r") as fin:
        lines = fin.readlines()

    rects1 = []
    for line in lines:
        data = line.split()
        y1, x1, y2, x2 = map(int, data)
        rects1.append([y1, x1, y2, x2])

print("Bounding boxes: ", rects1)

# Preprocess input
image = Image.open(args.input_image)
image.load()
sample = {'image': image, 'lines_boxes': rects1}
sample = Transform(sample)
image, boxes = sample['image'], sample['boxes']

image = image.to(device)
boxes = boxes.to(device)

# Feature extraction
with torch.no_grad():
    features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)

# Counting and adaptation
if not args.adapt:
    with torch.no_grad():
        output = regressor(features)
else:
    features.requires_grad = True
    optimizer = optim.Adam(regressor.parameters(), lr=args.learning_rate)

    for step in tqdm(range(args.gradient_steps), desc="Adaptation"):
        optimizer.zero_grad()
        output = regressor(features)
        lCount = args.weight_mincount * MincountLoss(output, boxes, use_gpu=device.type == "mps")
        lPerturbation = args.weight_perturbation * PerturbationLoss(output, boxes, sigma=8, use_gpu=device.type == "mps")
        Loss = lCount + lPerturbation
        Loss.backward()
        optimizer.step()

    features.requires_grad = False

# Visualization
rslt_file = "{}/{}_dot_out.png".format(args.output_dir, image_name)
# visualize_output_with_dots(image.detach().cpu(), output.detach().cpu(), boxes.cpu(), rslt_file, threshold=args.threshold)
visualize_output_with_accuracy_levels(
    image.detach().cpu(),
    output.detach().cpu(),
    boxes.cpu(),
    rslt_file,
    thresholds=(0.7, 0.65, 0.4),  # Adjust thresholds as needed
    min_distance=10  # Minimum distance for separating dots
)
print(f"===> The predicted count is: {output.sum().item():6.2f}")
high_count, medium_count, low_count, total_count = count_dots_with_adjusted_area(
    image.detach().cpu(),
    output.detach().cpu(),
    thresholds=(0.7, 0.65, 0.4),  # Density thresholds
    min_distance=10  # Minimum distance for separating dots
)

# python demo1.py -i aa.jpg -m data/pretrainedModels/FamNet_Save1.pth