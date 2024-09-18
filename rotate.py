# rotates and saves images to increase the dataset lol

import cv2 as cv
import face_recognition_utils.recognizer_powerhouse as rp
import os

def rotate_and_save_images(directory, celebrity_names):
    """Rotate images in each directory and save them with unique names."""
    for person in celebrity_names:
        path = os.path.join(directory, person)
        print("Folder path: ", path)
        label = celebrity_names.index(person)  # Generating int labels
        print(person)
        print("It's labelled as: ", label)
        
        # Ensure the directory exists
        if not os.path.isdir(path):
            print(f"Directory {path} does not exist. Skipping.")
            continue
        
        # Initialize the index for unique filenames
        i = 1
        
        for image_file in os.listdir(path):
            image_path = os.path.join(path, image_file)
            print("Image path: ", image_path)
            
            # Read the image
            img = cv.imread(image_path)
            
            # Check if image is successfully loaded
            if img is None:
                print(f"Failed to load image from {image_path}. Skipping.")
                continue
            
            # Rotate the image
            rotated_image = rp.image_rotation(img)
            
            # Save the rotated image with a unique name
            save_image(rotated_image, person, i)
            
            # Increment the index for the next image
            i += 1

def save_image(image, person, index):
    """Save the rotated image with a unique name."""
    # Define the filename with the person name and index
    filename = f"rotate_{person}_{index}.jpg"
    # Save the image in the same directory as the original
    save_path = os.path.join(DIR, person, filename)
    cv.imwrite(save_path, image)
    print(f"Saved rotated image as {save_path}")

# Define the directory and celebrity names
DIR = "./asset/Face_Recon_Dataset"
CELEBRITY_NAMES = ['50cent', 'Eminem', 'IceCube', 'Kanye', 'MichaelJackson']

# Call the function to process images
rotate_and_save_images(DIR, CELEBRITY_NAMES)
