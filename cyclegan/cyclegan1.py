import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from PIL import Image
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

# ====================
# 📌 Check GPU
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ====================
# 📌 Dataset Preparation
# ====================
class FaceToLegoDataset(Dataset):
    def __init__(self, og_folder, lg_folder, transform=None):
        self.og_folder = og_folder
        self.lg_folder = lg_folder
        self.transform = transform
        self.image_filenames = sorted(os.listdir(og_folder))  # Ensure matching images

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        og_image = Image.open(os.path.join(self.og_folder, img_name)).convert("RGB")
        lg_image = Image.open(os.path.join(self.lg_folder, img_name)).convert("RGB")

        if self.transform:
            og_image = self.transform(og_image)
            lg_image = self.transform(lg_image)

        return og_image, lg_image

# ====================
# 📌 Data Transformations
# ====================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Dataset
dataset = FaceToLegoDataset("./lego_ref_images/lg_cropped", "./lego_ref_images/og_cropped", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ====================
# 📌 Generator (U-Net)
# ====================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# ====================
# 📌 Discriminator (PatchGAN)
# ====================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ====================
# 📌 Initialize Models
# ====================
G_A2B = Generator().to(device)  # Real Face → LEGO
G_B2A = Generator().to(device)  # LEGO → Real Face
D_A = Discriminator().to(device)  # Discriminator for Real Faces
D_B = Discriminator().to(device)  # Discriminator for LEGO Faces

# Loss and Optimizers
criterion_GAN = nn.BCELoss()
criterion_Cycle = nn.L1Loss()  # Cycle consistency loss

optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))


# ====================
# 📌 Training Loop
# ====================
num_epochs = 10

for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, (real_A, real_B) in progress_bar:
        real_A, real_B = real_A.to(device), real_B.to(device)
        batch_size = real_A.size(0)

        # Train Generators
        optimizer_G.zero_grad()
        fake_B = G_A2B(real_A)  # A → B (Real Face → LEGO)
        fake_A = G_B2A(real_B)  # B → A (LEGO → Real Face)

        real_labels = torch.ones_like(D_B(real_B), device=device)
        fake_labels = torch.zeros_like(D_B(fake_B), device=device)

        # Generator loss
        loss_GAN_A2B = criterion_GAN(D_B(fake_B), real_labels)  # Loss for Real → LEGO
        loss_GAN_B2A = criterion_GAN(D_A(fake_A), fake_labels)  # Loss for LEGO → Real
        loss_cycle_A = criterion_Cycle(G_B2A(fake_B), real_A)  # Cycle consistency loss A → B → A
        loss_cycle_B = criterion_Cycle(G_A2B(fake_A), real_B)  # Cycle consistency loss B → A → B
        loss_G = loss_GAN_A2B + loss_GAN_B2A + 10 * (loss_cycle_A + loss_cycle_B)

        loss_G.backward()
        optimizer_G.step()

        # Train Discriminators
        optimizer_D_A.zero_grad()
        loss_D_A = criterion_GAN(D_A(real_A), real_labels) + criterion_GAN(D_A(fake_A.detach()), fake_labels)
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        loss_D_B = criterion_GAN(D_B(real_B), real_labels) + criterion_GAN(D_B(fake_B.detach()), fake_labels)
        loss_D_B.backward()
        optimizer_D_B.step()

        progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss G: {loss_G.item():.4f} Loss D: {(loss_D_A.item() + loss_D_B.item()):.4f}")

    # 🎯 Display & Save results every 10 epochs
    if (epoch + 1) % 5 == 0:
        save_image(fake_B[:8], f"./output/lego_epoch_{epoch+1}.png", normalize=True)  # LEGO generated from Real Faces
        save_image(fake_A[:8], f"./output/fake_faces_{epoch+1}.png", normalize=True)  # Reconstructed Real Faces from LEGO

print("Training complete! Models saved.")
