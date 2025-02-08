import os
import cv2
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

# Set input/output folders
input_folder = "./lego_ref_images/lg"  # Change to "lg" for LEGO dataset
output_folder = "./augmented/aug_lg"  
os.makedirs(output_folder, exist_ok=True)

# Function to apply Gaussian Blur manually (since torchvision doesn't support it in Torch 1.5)
def gaussian_blur(image, ksize=5):
    img_np = np.array(image)  # Convert PIL to NumPy
    img_blur = cv2.GaussianBlur(img_np, (ksize, ksize), 0)
    return Image.fromarray(img_blur)  # Convert back to PIL

# Data augmentation pipeline
def augment_image(image):
    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=20, shear=10),
        transforms.ToTensor()
    ])
    return transform(image)

num_augments = 20  # Number of new images per original

for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    image = Image.open(img_path).convert("RGB")
    
    for i in range(num_augments):
        transformed = augment_image(image)
        
        # Apply Gaussian Blur manually
        blurred_image = gaussian_blur(transformed.permute(1, 2, 0).mul(255).byte().numpy())  # Convert tensor → NumPy → PIL
        transformed = transforms.ToTensor()(blurred_image)  # Convert back to tensor
        
        save_image(transformed, os.path.join(output_folder, f"{filename.split('.')[0]}_aug{i}.jpg"))

print(f"✅ Data Augmentation Complete! Images saved in {output_folder}")

