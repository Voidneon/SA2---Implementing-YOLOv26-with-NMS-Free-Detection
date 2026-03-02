import cv2
# This line imports the OpenCV library (cv2) which provides computer vision tools for image processing.
# We'll use it to read images, apply Gaussian blur to license plates, and save the modified images.

import os
# This imports the os module which gives us functions to interact with the operating system.
# We use it to build file paths, check if directories exist, and handle file operations across different splits.

import glob
# The glob module helps us find all files matching a specific pattern (like all .txt files) in directories.
# This saves us from manually listing files and makes the code more efficient when processing multiple files.

def process_plates_and_shift_classes(base_path):
    # Define our main function that takes a base directory path as input.
    # This function will process all dataset splits, blur license plates, and renumber classes appropriately.
    
    # Process train, valid, and test directories
    splits = ['train', 'valid', 'test']
    # We create a list containing the three standard dataset splits used in YOLO object detection projects.
    # The function will iterate through each of these folders to process their contents.

    for split in splits:
        image_dir = os.path.join(base_path, split, 'images')
        label_dir = os.path.join(base_path, split, 'labels')
        # These lines construct the full paths to the images and labels folders for each split.
        # os.path.join intelligently combines path components with the correct OS-specific separators.

        if not os.path.exists(label_dir):
            continue
        # Check if the labels directory actually exists before trying to process it.
        # If it doesn't exist (like if we only have training data but no validation), we skip to the next split.

        label_files = glob.glob(os.path.join(label_dir, '*.txt'))
        print(f"Processing {len(label_files)} files in {split} set...")
        # Use glob to find all .txt annotation files in the labels directory.
        # Print a status message showing how many files we're processing to keep the user informed.

        for label_file in label_files:
            with open(label_file, 'r') as file:
                lines = file.readlines()
            # Open each annotation file in read mode and load all lines into a list.
            # The 'with' statement ensures the file is properly closed after reading.

            new_lines = []
            plates_to_blur = []
            # Initialize empty lists to store modified annotations and license plate coordinates.
            # plates_to_blur will collect bounding box data for plates we need to blur.

            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                # Split each line into parts and skip empty lines.
                # YOLO format typically has class_id followed by normalized coordinates (x_center, y_center, width, height)

                class_id = int(parts[0])
                # Extract the class ID as an integer for comparison.
                
                # Class 3 is license_plate
                if class_id == 3:
                    plates_to_blur.append([float(x) for x in parts[1:5]])
                    # If we find a license plate (class 3), we store its bounding box coordinates.
                    # Convert the string coordinates to floats for precise position calculations.
                else:
                    # Shift classes 4 (motorcycle) and 5 (truck) down by 1
                    if class_id > 3:
                        class_id -= 1
                    new_lines.append(f"{class_id} {' '.join(parts[1:])}\n")
                    # For non-plate objects, we first check if they have class IDs greater than 3.
                    # Since we're removing class 3 (license plates), we subtract 1 from higher classes to fill the gap.
                    # Then we rebuild the line with the updated class ID and add it to new_lines.

            # Blur the plates in the image
            if plates_to_blur:
                base_name = os.path.splitext(os.path.basename(label_file))[0]
                img_path = os.path.join(image_dir, base_name + '.jpg')
                # If we found any plates to blur, construct the corresponding image file path.
                # We extract the base filename without extension and add .jpg (assuming images are JPG format).
                
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w, _ = img.shape
                        # Read the image using OpenCV and check if it loaded successfully.
                        # Get image dimensions (height, width) to convert normalized coordinates to pixel coordinates.

                        for x_center, y_center, box_w, box_h in plates_to_blur:
                            x1 = max(0, int((x_center - box_w/2) * w))
                            y1 = max(0, int((y_center - box_h/2) * h))
                            x2 = min(w, int((x_center + box_w/2) * w))
                            y2 = min(h, int((y_center + box_h/2) * h))
                            # Convert normalized YOLO coordinates to pixel coordinates for the bounding box.
                            # Use max/min to ensure we don't go outside image boundaries.
                            # Calculate top-left (x1,y1) and bottom-right (x2,y2) corners.

                            plate_roi = img[y1:y2, x1:x2]
                            if plate_roi.size != 0:
                                blurred_roi = cv2.GaussianBlur(plate_roi, (51, 51), 30)
                                img[y1:y2, x1:x2] = blurred_roi
                                # Extract the region of interest (ROI) containing the license plate.
                                # Apply Gaussian blur with a 51x51 kernel and standard deviation 30.
                                # Replace the original plate region with the blurred version.

                        cv2.imwrite(img_path, img) # Overwrite original
                        # Save the modified image, overwriting the original file.
                        # This permanently applies the blur effect to license plates in the dataset.

            # Save the cleaned and shifted text file
            with open(label_file, 'w') as file:
                file.writelines(new_lines)
            # Write the updated annotations back to the original label file.
            # This removes all license plate annotations and renumbers the remaining classes.

    print("Pipeline Complete: Plates blurred, class dropped, IDs shifted.")
    # Print a completion message to confirm the entire process finished successfully.
    # This gives clear feedback that the dataset transformation is complete.

# Run the script (assuming it's in the same folder as train/valid/test)
process_plates_and_shift_classes('.')
# Execute the function with the current directory as the base path.
# The '.' tells Python to look for train/valid/test folders in the same location as the script.