import os

# Define the path to the 'img' folder
img_folder = "./data/indt-objects-V4 copy"

# Get a sorted list of all JPG files (ignoring hidden/system files)
files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith('.jpg')])

# Debug: Print the total number of files found
print(f"Total JPG files found: {len(files)}")

# Rename each file to a zero-padded three-digit number followed by .jpg (starting from 001)
for index, file_name in enumerate(files, start=1):
    old_file_path = os.path.join(img_folder, file_name)
    new_file_name = f"{index:03d}.jpg"  # Formats index as 001, 002, ..., 999
    new_file_path = os.path.join(img_folder, new_file_name)

    # Debug: Print each renaming step
    print(f"Renaming: {file_name} -> {new_file_name}")

    # Rename the file
    os.rename(old_file_path, new_file_path)

print("All files in the folder have been renamed.")
