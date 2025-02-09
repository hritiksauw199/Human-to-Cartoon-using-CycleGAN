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
dataset = FaceToLegoDataset("./lego_ref_images/og_cropped", "./lego_ref_images/lg_cropped", transform=transform)
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
G_h = Generator().to(device)  # Real Face → LEGO
G_z = Generator().to(device)  # LEGO → Real Face
D_h = Discriminator().to(device)  # Discriminator for Real Faces
D_z = Discriminator().to(device)  # Discriminator for LEGO Faces

# Loss and Optimizers
criterion_GAN = nn.BCELoss()
criterion_Cycle = nn.L1Loss()  # Cycle consistency loss

#optimizer_G = optim.Adam(itertools.chain(G_z.parameters(), G_h.parameters()), lr=0.0002, betas=(0.5, 0.999))
#optimizer_D_z = optim.Adam(D_z.parameters(), lr=0.0002, betas=(0.5, 0.999))
#optimizer_D_h = optim.Adam(D_h.parameters(), lr=0.0002, betas=(0.5, 0.999))

opt_disc = optim.Adam(
        list(D_h.parameters()) + list(D_z.parameters()),
        lr=0.0002,
        betas=(0.5, 0.999),
    )

opt_gen = optim.Adam(
        list(G_z.parameters()) + list(G_h.parameters()),
        lr=0.0002,
        betas=(0.5, 0.999),
    )
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

# ====================
# 📌 Training Loop
# ====================
num_epochs = 10

for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for i, (zebra, horse) in progress_bar:
        zebra, horse = zebra.to(device), horse.to(device)
        batch_size = zebra.size(0)

        # Discr
        fake_horse = G_h(zebra)  # A → B (Real Face → LEGO)
        D_H_real = D_h(horse)
        D_H_fake = D_h(horse.detach())
        loss_D_B = criterion_GAN(D_H_real, torch.ones_like(D_H_real)) + criterion_GAN(D_H_fake, torch.zeros_like(D_H_fake))


        fake_zebra = G_z(horse)
        D_Z_real = D_z(zebra)
        D_Z_fake = D_z(fake_zebra.detach())
        loss_D_A = criterion_GAN(D_Z_real, torch.ones_like(D_Z_real)) + criterion_GAN(D_Z_fake, torch.ones_like(D_H_fake))
        
        D_loss = (loss_D_A + loss_D_B) /2
        
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        #optimizer_D_A.zero_grad()
        #loss_D_A.backward()
        #optimizer_D_A.step()



        # Train Generators

        # Generator loss
        D_H_fake = D_h(fake_horse)
        D_Z_fake = D_z(fake_zebra)
        loss_G_H = criterion_GAN(D_H_fake, torch.ones_like(D_H_fake))  # Loss for Real → LEGO
        loss_G_Z = criterion_GAN(D_Z_fake, torch.ones_like(D_Z_fake))  # Loss for LEGO → Real
        
        # cycle losses
        loss_cycle_zebra = criterion_Cycle(zebra, G_z(fake_horse))  # Cycle consistency loss A → B → A
        loss_cycle_horse = criterion_Cycle(horse, G_h(fake_zebra))  # Cycle consistency loss B → A → B
        
        #total loss
        loss_G = loss_G_H + loss_G_Z + 10 * (loss_cycle_zebra + loss_cycle_horse)



        #optimizer_G.zero_grad()
        #loss_G.backward()
        #optimizer_G.step()
        opt_gen.zero_grad()
        g_scaler.scale(loss_G).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss G: {loss_G.item():.4f} Loss D: {(loss_D_A.item() + loss_D_B.item()):.4f}")

    # 🎯 Display & Save results every 10 epochs
    if (epoch + 1) % 5 == 0:
        save_image(fake_horse[:8], f"./output/lego_epoch_{epoch+1}.png", normalize=True)  # LEGO generated from Real Faces
        save_image(fake_zebra[:8], f"./output/reconstructed_epoch_{epoch+1}.png", normalize=True)  # Reconstructed Real Faces from LEGO

print("Training complete! Models saved.")
