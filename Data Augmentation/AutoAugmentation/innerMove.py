import torch
from torchvision import transforms
from PIL import Image
import os

class AdvancedAutoAugment(torch.nn.Module):
    def __init__(self):
        super(AdvancedAutoAugment, self).__init__()
        # Define the sequence of image transformations
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.RandomHorizontalFlip(),  # Apply random horizontal flipping
            transforms.RandomRotation(30),  # Rotate the image randomly up to 30 degrees
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33)),  # Randomly resize and crop the image
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly jitter color
            transforms.AutoAugment()  # Apply AutoAugment policies
        ])

    def forward(self, img):
        # Apply the transformations to the image
        return self.transforms(img)

# Setup for loading and saving images
image_path = "C:\Your\Path\To\Images"
save_directory = 'C:\Your\Path\To\Images'
os.makedirs(save_directory, exist_ok=True)  # Ensure the output directory exists
image = Image.open(image_path)  # Open the image

# Use the Advanced AutoAugment class for data augmentation
advanced_auto_augment = AdvancedAutoAugment()

# Save each augmented image multiple times
for i in range(30):  # Generate and save 30 different augmented images
    augmented_image = advanced_auto_augment(image)
    augmented_image.save(os.path.join(save_directory, f'augmented_image_{i}.jpg'))  # Save the augmented images

print("Augmented images have been successfully saved.")
