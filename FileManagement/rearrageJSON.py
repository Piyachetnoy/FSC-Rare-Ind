import json
import numpy as np

# Load data from JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Save data to JSON file
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        json.dump(data, file, indent=4, default=convert)

def rearrange_coordinates(box_coordinates):
    rearranged_boxes = []
    for box in box_coordinates:
        # Convert to numpy array for easier manipulation
        box = np.array(box)
        # Sort by y-coordinate first (ascending), then by x-coordinate (ascending)
        sorted_box = box[np.lexsort((box[:, 0], box[:, 1]))]
        # Bottom-left
        bottom_left = sorted_box[0]
        # Bottom-right
        bottom_right = sorted_box[1] if sorted_box[1][0] > sorted_box[0][0] else sorted_box[0]
        # Top-right
        top_right = sorted_box[3] if sorted_box[3][0] > sorted_box[2][0] else sorted_box[2]
        # Top-left
        top_left = sorted_box[3] if sorted_box[3][0] < sorted_box[2][0] else sorted_box[2]

        # Rearrange to [bottom left, bottom right, top right, top left]
        rearranged_boxes.append([list(bottom_left), list(bottom_right), list(top_right), list(top_left)])
    return rearranged_boxes

# Path to the JSON file
file_path = './data_final/annotations.json'

# Load the JSON data
data = load_json(file_path)

# Process each key in the JSON data
for key, value in data.items():
    if "box_examples_coordinates" in value:
        # Rearrange the coordinates
        value["box_examples_coordinates"] = rearrange_coordinates(value["box_examples_coordinates"])

# Save the updated data back to the JSON file
save_json(data, file_path)

# Print confirmation
print("Updated JSON data saved to", file_path)
