import os
import random
import json

# Define the path to the image folder
img_folder = "./data-V2/indt-objects-V5"

# Get a sorted list of all JPG files (ignoring hidden/system files)
files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith('.jpg') and not f.lower().startswith('.')])

# Shuffle the file list randomly
random.shuffle(files)

# Calculate the split sizes
total_files = len(files)
train_size = int(total_files * 0.70)
val_size = int(total_files * 0.20)
test_size = total_files - train_size - val_size

# Split the files into train, val, and test sets
train_files = files[:train_size]
val_files = files[train_size:train_size + val_size]
test_files = files[train_size + val_size:]

# Create the JSON object
data_splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

# Save the splits to a JSON file
output_file = "./data-V2/Train_Test_Val.json"
with open(output_file, "w") as json_file:
    json.dump(data_splits, json_file, indent=4)

print(f"Image file names have been distributed and saved to {output_file}.")
