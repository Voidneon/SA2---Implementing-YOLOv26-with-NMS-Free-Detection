import cv2
# This line imports the OpenCV library which provides computer vision and image processing functions.
# We'll use it to read images, flip them horizontally, and save the augmented versions.

import os
# This imports the os module for operating system interactions like building file paths and checking directories.
# It helps us navigate the folder structure where our training images and labels are stored.

import glob
# The glob module allows us to find all files that match a specific pattern (like all .txt files in a folder).
# This makes it easy to process multiple annotation files without having to list them manually.

def augment_minority_classes(train_path):
    # Define our main function that takes the path to the training folder as input.
    # This function will find images containing minority classes and create flipped versions to balance our dataset.

    image_dir = os.path.join(train_path, 'images')
    label_dir = os.path.join(train_path, 'labels')
    # Build the complete paths to the images and labels subfolders within the training directory.
    # os.path.join ensures the correct path separators are used for the current operating system.
    
    # Target classes: bus(0), jeepney(2), motorcycle(3), truck(4)
    minority_classes = [0, 2, 3, 4] 
    # Define a list of class IDs that are underrepresented in our dataset.
    # These are the classes we want to create more samples for through augmentation.
    # Comments help us remember which vehicle type corresponds to each number.
    
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    print(f"Scanning {len(label_files)} training files for augmentation...")
    # Find all text files in the labels directory that contain our bounding box annotations.
    # Print a status message showing how many files we'll check through.
    
    augmented_count = 0
    # Initialize a counter to keep track of how many new augmented images we create.
    # This helps us verify that our augmentation process is working correctly.

    for label_file in label_files:
        # Loop through each annotation file in the training labels folder.
        # We'll check each one to see if it contains any minority class objects.
        
        with open(label_file, 'r') as file:
            lines = file.readlines()
        # Open the current label file in read mode and load all lines into a list.
        # Each line represents one object in the image with its class ID and bounding box coordinates.

        # Check if the frame contains any minority class
        needs_augmentation = False
        # Set up a flag to track whether this image needs augmentation.
        # We start with False and change to True if we find any minority class.
        
        for line in lines:
            if int(line.split()[0]) in minority_classes:
                needs_augmentation = True
                break
        # Go through each object in the image and check its class ID.
        # If we find any object whose class is in our minority_classes list, set the flag to True.
        # Use break to stop checking once we find one minority object (that's all we need to know).
                
        if needs_augmentation:
            # If this image contains at least one minority class object, proceed with augmentation.
            
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            img_path = os.path.join(image_dir, base_name + '.jpg')
            # Extract the base filename without extension and build the path to the corresponding image.
            # We assume images are in JPG format (common for object detection datasets).
            
            if os.path.exists(img_path):
                # Check if the image file actually exists before trying to read it.
                # This prevents errors if an annotation file exists without a matching image.
                
                # 1. Flip the Image
                img = cv2.imread(img_path)
                # Read the original image using OpenCV's imread function.
                # This loads the image as a numpy array that we can manipulate.
                
                flipped_img = cv2.flip(img, 1) # 1 means horizontal flip
                # Apply horizontal flip to create a mirror image of the original.
                # The parameter 1 specifies horizontal flipping (0 would be vertical flip).
                
                new_img_path = os.path.join(image_dir, f"{base_name}_aug_flip.jpg")
                cv2.imwrite(new_img_path, flipped_img)
                # Create a new filename by adding "_aug_flip" to the original base name.
                # Save the flipped image to the images folder with this new name.
                
                # 2. Flip the Bounding Boxes
                new_lines = []
                # Initialize an empty list to store our transformed annotations.
                
                for line in lines:
                    parts = line.strip().split()
                    cls_id = parts[0]
                    # Split each annotation line into its components.
                    # YOLO format is: class_id x_center y_center width height (all normalized 0-1)
                    
                    # Invert the X center coordinate
                    new_x_center = 1.0 - float(parts[1])
                    # When we flip an image horizontally, the X coordinate needs to be mirrored.
                    # If an object was at 0.2 from the left, after flipping it will be at 0.8 from the left.
                    # Formula: new_x = 1 - old_x (since coordinates are normalized between 0 and 1)
                    
                    new_lines.append(f"{cls_id} {new_x_center:.6f} {' '.join(parts[2:])}\n")
                    # Rebuild the annotation line with the new X coordinate.
                    # Keep the Y coordinate, width, and height exactly the same (they don't change with horizontal flip).
                    # Format the new X coordinate with 6 decimal places for precision.
                
                new_label_path = os.path.join(label_dir, f"{base_name}_aug_flip.txt")
                with open(new_label_path, 'w') as new_file:
                    new_file.writelines(new_lines)
                # Create a new annotation file with the same "_aug_flip" suffix as the image.
                # Write all the transformed annotations to this new file.
                    
                augmented_count += 1
                # Increment our counter each time we successfully create an augmented sample.
                # This gives us a final count of how many new images we generated.

    print(f"Augmentation complete! Generated {augmented_count} new minority-balanced frames.")
    # Print a completion message showing how many augmented images were created.
    # This provides immediate feedback about the success of our data balancing effort.

# Run the script ONLY on the training folder
augment_minority_classes('./train')
# Call our function with the path to the training folder as the argument.
# Only augment training data (never validation or test data).