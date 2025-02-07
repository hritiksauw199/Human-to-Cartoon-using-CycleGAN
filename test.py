# test.py

import torch
from PIL import Image
import torchvision.transforms as transforms
from trycgan import Generator  # Assuming Generator class is defined in model.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained models
G_h = Generator().to(device)
G_z = Generator().to(device)
G_h.load_state_dict(torch.load("generator_face_to_lego.pth"))
G_z.load_state_dict(torch.load("generator_lego_to_face.pth"))
G_h.eval()
G_z.eval()

# Define image transformation for testing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Load a new image and generate LEGO version
image_path = "image.png"  # Replace with the path to the new face image
new_image = load_and_preprocess_image(image_path, transform)

# Generate LEGO image
with torch.no_grad():  # Disable gradient computation for inference
    lego_image = G_h(new_image)

# Post-process the output and convert to PIL for saving/display
lego_image = lego_image.squeeze(0)  # Remove batch dimension
lego_image = lego_image.cpu().clamp(0, 1)  # Ensure pixel values are in [0, 1]

# Convert tensor to PIL Image for display
lego_image_pil = transforms.ToPILImage()(lego_image)
lego_image_pil.show()  # Display the generated LEGO image
lego_image_pil.save("generated_lego_image.jpg")  # Optionally save the image

print("Testing complete!")
