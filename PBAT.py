import cv2
import numpy as np
import json
import os

# Global variable for annotations
annotations = {}

# Current annotation details
rect_points = []
dot_points = []
undo_stack = []

# Mouse callback function for drawing annotations
def mouse_callback(event, x, y, flags, param):
    global rect_points, dot_points, undo_stack

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_points.append((x, y))
        undo_stack.append(("rect", len(rect_points) - 1))
    elif event == cv2.EVENT_RBUTTONDOWN:
        dot_points.append((x, y))
        undo_stack.append(("dot", len(dot_points) - 1))

# Undo the last action
def undo_action():
    global rect_points, dot_points, undo_stack

    if undo_stack:
        last_action, index = undo_stack.pop()
        if last_action == "rect" and index < len(rect_points):
            rect_points.pop(index)
        elif last_action == "dot" and index < len(dot_points):
            dot_points.pop(index)

# Function to save annotations for a single image
def save_annotations(filename, original_height, original_width):
    global rect_points, dot_points

    ratio_h = 384/64
    ratio_w = 384/64

    # Create rectangular box coordinates from rect_points
    box_examples_coordinates = []
    for i in range(0, len(rect_points), 2):
        if i + 1 < len(rect_points):
            x1, y1 = rect_points[i]
            x2, y2 = rect_points[i + 1]
            box = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            box_examples_coordinates.append(box)

    annotation = {
        "H": original_height,
        "W": original_width,
        "box_examples_coordinates": box_examples_coordinates,
        "box_examples_path": [f"/nfs/bigneuron/viresh/FSC_NewDataOnly/box_examples/{filename}_{i}.jpg" for i in range(len(box_examples_coordinates))],
        "density_path": f"/nfs/bigneuron/viresh/FSC_NewDataOnly/gt_density_map_adaptive_384_VarV2/{filename}.npy",
        "density_path_fixed": f"/nfs/bigneuron/viresh/FSC_NewDataOnly/gt_density_map_fixed/{filename}.npy",
        "img_path": f"/nfs/bigneuron/viresh/FSC_NewDataOnly/images_384_VarV2/{filename}.jpg",
        "points": dot_points,
        "r": [30] * 2,
        "ratio_h": ratio_h,
        "ratio_w": ratio_w
    }

    return annotation

# Main function for annotating multiple images
def annotate_images(image_folder):
    global rect_points, dot_points, undo_stack, annotations

    for image_file in os.listdir(image_folder):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load the image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        original_height, original_width = image.shape[:2]

        # Set up the window and callback
        cv2.namedWindow("PBAT (Point and Box Annotation Tool)")
        cv2.setMouseCallback("PBAT (Point and Box Annotation Tool)", mouse_callback)

        while True:
            display_image = image.copy()
            cv2.putText(display_image, image_file, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            # Draw rectangular box points as small circles
            for point in rect_points:
                cv2.circle(display_image, point, 5, (255, 255, 0), -1)  # Highlight the clicked points

            # Draw rectangular boxes
            for i in range(0, len(rect_points), 2):
                if i + 1 < len(rect_points):
                    cv2.rectangle(display_image, rect_points[i], rect_points[i + 1], (0, 255, 0), 2)

            # Draw dots
            for dot in dot_points:
                cv2.circle(display_image, dot, 4, (0, 0, 255), -1)

            cv2.imshow("PBAT (Point and Box Annotation Tool)", display_image)

            key = cv2.waitKey(1)
            if key == ord('s'):  # Save annotations
                annotations[image_file] = save_annotations(image_file[:-4], original_height, original_width)
                rect_points = []
                dot_points = []
                undo_stack = []
                break
            elif key == ord('n'):  # Skip current image without saving
                rect_points = []
                dot_points = []
                undo_stack = []
                break
            elif key == ord('q'):  # Quit annotation
                cv2.destroyAllWindows()
                return
            elif key == 26:  # Ctrl+Z for undo (ASCII code for Ctrl+Z is 26)
                undo_action()

        cv2.destroyAllWindows()

# Save annotations to a JSON file
def save_to_json(output_file):
    global annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)

# Example usage
if __name__ == "__main__":
    image_folder = "data/indt-objects-V4"
    output_file = "annotations10.json"

    annotate_images(image_folder)
    save_to_json(output_file)
    print(f"Annotations saved to {output_file}")
