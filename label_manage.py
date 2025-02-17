import os

# Define the paths to the images and labels folders
images_folder = "data/indt-objects-V1"
labels_folder = "data/labels"

# Get the set of image filenames without extension
image_filenames = {os.path.splitext(f)[0] for f in os.listdir(images_folder) if f.endswith(".jpg")}

# Iterate through the label files
for label_file in os.listdir(labels_folder):
    if label_file.endswith(".txt"):
        label_name = os.path.splitext(label_file)[0]  # Get filename without extension
        label_path = os.path.join(labels_folder, label_file)
        
        # Check if there is no corresponding image
        if label_name not in image_filenames:
            print(f"Deleting unmatched label file: {label_file}")
            os.remove(label_path)