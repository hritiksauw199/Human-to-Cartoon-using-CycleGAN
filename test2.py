import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from trial import Generator

# ====================
# 📌 Load the Saved Model Weights
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the generator models
G_h = Generator().to(device)  # Real Face → LEGO
G_z = Generator().to(device)  # LEGO → Real Face

# Load the saved weights
G_h.load_state_dict(torch.load("generator_face_to_lego.pth"))
G_z.load_state_dict(torch.load("generator_lego_to_face.pth"))
G_h.eval()
G_z.eval()

# ====================
# 📌 Image Preprocessing (same as training)
# ====================
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize like training
])

def preprocess_image(image_path):
    # Load image
    img = Image.open(image_path).convert("RGB")
    
    # Apply transformations
    img = transform(img)
    
    # Add batch dimension (1 image in batch)
    img = img.unsqueeze(0).to(device)
    
    return img

# ====================
# 📌 Test the Model on a New Image
# ====================
def test_model(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Pass through the generator to get the output
    with torch.no_grad():  # No need to compute gradients for testing
        generated_image = G_h(img)  # Real Face → LEGO (you can swap with G_z if testing the reverse)
    
    # Post-process the image
    generated_image = generated_image.squeeze(0).cpu()  # Remove batch dimension
    generated_image = generated_image * 0.5 + 0.5  # Denormalize to [0, 1]
    
    # Convert to a PIL Image for display
    generated_image = transforms.ToPILImage()(generated_image)
    
    # Display the result
    plt.imshow(generated_image)
    plt.axis("off")
    plt.show()

# Example: Test with a new image
test_image_path = "image.png"  # Replace with your test image path
test_model(test_image_path)
