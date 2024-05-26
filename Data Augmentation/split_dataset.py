import os
import shutil
import random

# Define source and destination directories
source_dir = 'C:\\Users\\nilay\\Desktop\\Dataset\\MainDataset\\Training'
destination_dir = 'C:\\Users\\nilay\\Desktop\\Dataset\\MainDataset\\Test'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Function to move 10% of images from each subdirectory
def move_images(source, dest):
    for subdir in os.listdir(source):
        source_subdir = os.path.join(source, subdir)
        dest_subdir = os.path.join(dest, subdir)
        
        # Create destination subdirectory if it doesn't exist
        os.makedirs(dest_subdir, exist_ok=True)
        
        # List all files in the source subdirectory
        files = [f for f in os.listdir(source_subdir) if os.path.isfile(os.path.join(source_subdir, f))]
        
        # Calculate 10% of the files
        num_files_to_move = max(1, len(files) // 10)
        
        # Select random files to move
        files_to_move = random.sample(files, num_files_to_move)
        
        # Move selected files
        for file in files_to_move:
            shutil.move(os.path.join(source_subdir, file), os.path.join(dest_subdir, file))
            print(f"Moved {file} to {dest_subdir}")

# Execute the function
move_images(source_dir, destination_dir)
