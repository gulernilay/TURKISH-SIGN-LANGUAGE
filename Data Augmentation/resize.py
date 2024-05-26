import os
from PIL import Image

# Define source and destination directories
# Input and output directories
input_directory = "/path/to/input/directory"
destination_dir = "/path/to/output/directory"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Function to resize images from each subdirectory
def resize_images(source, dest):
    # Iterate through each subdirectory in the source directory
    for subdir in os.listdir(source):
        source_subdir = os.path.join(source, subdir)
        dest_subdir = os.path.join(dest, subdir)
        
        # Create the destination subdirectory if it doesn't exist
        os.makedirs(dest_subdir, exist_ok=True)
        
        # Filter for image files in the source subdirectory
        files = [f for f in os.listdir(source_subdir) if os.path.isfile(os.path.join(source_subdir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Resize and save each image
        for file in files:
            file_path = os.path.join(source_subdir, file)
            with Image.open(file_path) as img:
                # Resize the image using the LANCZOS filter for high-quality downsampling
                img_resized = img.resize((224, 224), Image.LANCZOS)
                
                # Save the resized image to the destination directory
                img_resized.save(os.path.join(dest_subdir, file))
                
                # Optionally, print a message confirming the save
                print(f"Resized and saved {file} to {dest_subdir}")

# Call the function to resize and save images
resize_images(input_directory, destination_dir)
