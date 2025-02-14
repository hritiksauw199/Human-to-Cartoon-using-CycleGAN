import torch
from PIL import Image
import torchvision.transforms as transforms
from cyclegan.trial import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for models (so they don't reload unnecessarily)
G_h = None
G_z = None

def load_models():
    """Load the models only if they are not already loaded"""
    global G_h, G_z
    if G_h is None or G_z is None:  # Load only if models are not loaded
        G_h = Generator().to(device)
        G_z = Generator().to(device)
        G_h.load_state_dict(torch.load("generator_face_to_lego.pth"))
        G_z.load_state_dict(torch.load("generator_lego_to_face.pth"))
        G_h.eval()
        G_z.eval()
        print("Models loaded successfully!")
    else:
        print("Models already loaded, skipping reloading.")

# Call this function once at the start
load_models()

# Define image transformation for testing
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Function to generate LEGO-style image from a real face
def generate_lego_image(image_path, save_path="generated_lego_image_new.jpg"):
    global G_h  # Use the loaded model
    if G_h is None:
        print("Error: Model not loaded!")
        return

    image = load_and_preprocess_image(image_path)

    with torch.no_grad():  # Disable gradient computation for inference
        lego_image = G_h(image)

    # Post-process the output and convert to PIL for saving/display
    lego_image = lego_image.squeeze(0)  # Remove batch dimension
    lego_image = lego_image.cpu().clamp(0, 1)  # Ensure pixel values are in [0, 1]
    
    lego_image_pil = transforms.ToPILImage()(lego_image)
    lego_image_pil.show()  # Display the generated LEGO image
    lego_image_pil.save(save_path)  # Save the image

    print(f"Generated image saved at: {save_path}")

# Example usage
image_path = "image.png"  # Replace with actual image path
generate_lego_image(image_path)
