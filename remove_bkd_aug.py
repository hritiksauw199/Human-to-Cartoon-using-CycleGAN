import os
from PIL import Image
from rembg import remove

# Input and output directories
input_folder = "./dataset/cartoons_aug/"  # Folder containing images
output_folder = "./dataset/cartoon_aug/"  # Folder to save processed images

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg")):  # Process only image files
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)  # Save with the same name

        # Load and remove background
        input_image = Image.open(input_path)
        output_image = remove(input_image).convert('RGB')
        
        # Save the processed image
        output_image.save(output_path)
        print(f"Processed: {filename}")

print("Background removal complete for all images!")
