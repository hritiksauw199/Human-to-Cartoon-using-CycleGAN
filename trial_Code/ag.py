import os
import torch
import torchvision.transforms as T
from PIL import Image

# Input and output directories
input_folder = "./dataset/cartoons_without_glasses"  # Source dataset
output_folder = './dataset/cartoons_aug'  # Augmented dataset

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define augmentations
augmentations = T.Compose([
    T.RandomHorizontalFlip(p=1), 
     T.RandomRotation(30), # Horizontal Flip
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Brightness & Color Jitter
    T.RandomAffine(degrees=15, scale=(0.8, 1.2)),  # Zoom In/Out & Rotation
    T.ToTensor(),  # Convert to Tensor  # Gaussian Noise
    T.ToPILImage()  # Convert back to PIL
])

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Ensure valid image format
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)  # Keep same filename

        # Load and augment image
        image = Image.open(input_path).convert("RGB")  # Ensure RGB mode
        augmented_image = augmentations(image)

        # Save augmented image
        augmented_image.save(output_path)
        print(f"Augmented and saved: {filename}")

print("Data augmentation complete! Augmented images saved in 'low' dataset.")
