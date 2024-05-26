import os
from PIL import Image
import torch
from torchvision import transforms

class ResizeAndRotate(torch.nn.Module):
    def __init__(self, angle):
        super(ResizeAndRotate, self).__init__()
        # Define a series of image transformations
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
            transforms.RandomRotation(angle)  # Randomly rotate images by the specified angle
        ])

    def forward(self, img):
        # Apply the defined transformations to the input image
        return self.transforms(img)

# Input and output directories
input_directory = "/path/to/input/directory"
output_directory = "/path/to/output/directory"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Define the list of rotation angles
rotation_angles = [0, 30, 90, 135, 210, 270, 330]  

# Process each file in the input directory
for filename in os.listdir(input_directory):
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
        image_path = os.path.join(input_directory, filename)
        image = Image.open(image_path)

        # Rotate and resize each image using defined angles
        for i, angle in enumerate(rotation_angles):
            resize_and_rotate = ResizeAndRotate(angle)
            augmented_image = resize_and_rotate(image)

            # Ensure the image is in RGB format
            if augmented_image.mode != 'RGB':
                augmented_image = augmented_image.convert('RGB')

            # Save the processed image with a descriptive name
            augmented_image.save(os.path.join(output_directory, f'augmented_{filename[:-4]}_angle_{angle}.jpg'))

# Notification when processing is complete
print("All augmented images have been successfully saved.")
