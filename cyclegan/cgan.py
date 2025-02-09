import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class FaceToLegoDataset(Dataset):
    def __init__(self, og_folder, lg_folder, transform=None):
        self.og_folder = og_folder
        self.lg_folder = lg_folder
        self.transform = transform
        self.image_filenames = sorted(os.listdir(og_folder))
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        og_image = Image.open(os.path.join(self.og_folder, img_name)).convert("RGB")  # Ensuring RGB format
        lg_image = Image.open(os.path.join(self.lg_folder, img_name)).convert("RGB")  # Ensuring RGB format
        
        if self.transform:
            og_image = self.transform(og_image)  # Transform to tensor
            lg_image = self.transform(lg_image)  # Transform to tensor
        
        return og_image, lg_image

# Define the transform that converts images to tensor and normalizes them
transform = transforms.Compose([
    transforms.Resize((128, 128)),           # Resize images to a fixed size
    transforms.ToTensor(),                   # Convert images to tensor format [C, H, W]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with mean and std for 3 channels
])

# Use the dataset class and apply the transform
dataset = FaceToLegoDataset("./augmented/aug_og", "./augmented/aug_lg", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define Residual Block for Generator
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip Connection

# Define the Generator (Deeper CNN)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

# Define the Discriminator (More Layers)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1),  # Adjusted for deeper layers
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Feature Loss using VGG (Perceptual Loss)
class VGGFeatureLoss(nn.Module):
    def __init__(self):
        super(VGGFeatureLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:8].to(device).eval()  # Use first few layers
        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights
        self.vgg = vgg

    def forward(self, x, y):
        return torch.mean((self.vgg(x) - self.vgg(y)) ** 2)  # Mean Squared Error

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
feature_loss = VGGFeatureLoss().to(device)

# Loss and Optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (real_faces, real_legos) in progress_bar:
        real_faces, real_legos = real_faces.to(device), real_legos.to(device)
        batch_size = real_faces.size(0)
        
        # Create real and fake labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train Generator
        optimizer_G.zero_grad()
        fake_legos = generator(real_faces)
        fake_preds = discriminator(fake_legos)
        g_loss = criterion(fake_preds, real_labels) + 0.1 * feature_loss(fake_legos, real_legos)  # Added VGG Loss
        g_loss.backward()
        optimizer_G.step()
        
        # Train Discriminator
        optimizer_D.zero_grad()
        real_preds = discriminator(real_legos)
        d_real_loss = criterion(real_preds, real_labels)
        fake_preds = discriminator(fake_legos.detach())
        d_fake_loss = criterion(fake_preds, fake_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")
    
    # Save sample output
    if (epoch + 1) % 100 == 0:
        save_image(make_grid(fake_legos[:16], normalize=True), f"output_epoch_{epoch+1}.png")

print("✅ Training complete!")
