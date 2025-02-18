import os
import shutil

# Define the paths for the source folders and destination folder
base_path = "data"
folders = ["test", "train", "valid"]
destination_folder = os.path.join(base_path, "images_384_VarV2")

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through each folder and move .jpg files to the destination folder
for folder in folders:
    source_folder = os.path.join(base_path, folder)
    if os.path.exists(source_folder):
        for file_name in os.listdir(source_folder):
            if file_name.lower().endswith(".jpg"):
                source_file = os.path.join(source_folder, file_name)
                destination_file = os.path.join(destination_folder, file_name)

                # Move the file
                shutil.move(source_file, destination_file)
                print(f"Moved: {source_file} -> {destination_file}")
    else:
        print(f"Folder does not exist: {source_folder}")

print("All .jpg files have been moved to the folder.")
