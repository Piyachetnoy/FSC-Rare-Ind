import os

# Define the path to the 'img' folder
img_folder = "data/images_384_VarV2"

# Get a sorted list of all files in the 'img' folder
files = sorted(os.listdir(img_folder))

# Rename each file to a number followed by .jpg (starting from 1)
for index, file_name in enumerate(files, start=1):  # Start index from 1
    old_file_path = os.path.join(img_folder, file_name)
    new_file_name = f"{index}.jpg"
    new_file_path = os.path.join(img_folder, new_file_name)

    # Rename the file
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {old_file_path} -> {new_file_path}")

print("All files in the folder have been renamed.")
