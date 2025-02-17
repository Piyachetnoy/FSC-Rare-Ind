import os
import random
import json

# Define the path to the image folder
img_folder = "dataTest0/images_384_VarV2"

# Get a list of all image file names in the folder
files = os.listdir(img_folder)

# Shuffle the file list randomly
random.shuffle(files)

# Calculate the split sizes
total_files = len(files)
train_size = int(total_files * 0.75)
val_size = int(total_files * 0.15)
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
output_file = "dataTest0/Train_Test_Val_FSC_147.json"
with open(output_file, "w") as json_file:
    json.dump(data_splits, json_file, indent=4)

print(f"Image file names have been distributed and saved to {output_file}.")
