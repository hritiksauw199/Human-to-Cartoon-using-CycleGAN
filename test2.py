import torch
from torchvision.utils import save_image
import os
from PIL import Image
import torchvision.transforms as transforms
from trial2 import Generator

# ====================
# 📌 Load the Trained Generator
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
G.load_state_dict(torch.load("generator_face_to_lego.pth"))  # Load the trained model
G.eval()  # Set the model to evaluation mode

# ====================
# 📌 Test Data Preparation (For example, single test image)
# ====================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to load and preprocess a test image
def load_test_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
    return image

# Example of testing with a single image (adjust the path as needed)
test_image_path = "image.png"  # Replace with your test image path
test_image = load_test_image(test_image_path)

# ====================
# 📌 Generate LEGO-style Image
# ====================
with torch.no_grad():  # Disable gradient calculation for testing
    generated_image = G(test_image)  # Pass the test image through the generator

# ====================
# 📌 Save or Visualize the Result
# ====================
save_image(generated_image, "generated_lego_image.png", normalize=True)  # Save the output image

# Or if you want to visualize the image using matplotlib
import matplotlib.pyplot as plt
generated_image = generated_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
generated_image = (generated_image + 1) / 2  # Denormalize the image

plt.imshow(generated_image)
plt.axis('off')  # Turn off axis
plt.show()
