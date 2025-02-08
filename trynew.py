import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from PIL import Image
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
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ====================
# 📌 Optimized Convolutional and Residual Blocks
# ====================
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_downsampling: bool = True, add_activation: bool = True, **kwargs):
        super().__init__()
        if is_downsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )

    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvolutionalBlock(channels, channels, add_activation=True, kernel_size=3, padding=1),
            ConvolutionalBlock(channels, channels, add_activation=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

# ====================
# 📌 Generator (U-Net with Residual Blocks and Instance Normalization)
# ====================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder: Increase channel depth
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            ConvolutionalBlock(64, 128, is_downsampling=True, kernel_size=3, padding=1),
            ConvolutionalBlock(128, 256, is_downsampling=True, kernel_size=3, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Decoder: Decrease channel depth
        self.decoder = nn.Sequential(
            ConvolutionalBlock(256, 128, is_downsampling=False, kernel_size=3, padding=1),
            ConvolutionalBlock(128, 64, is_downsampling=False, kernel_size=3, padding=1),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


# ====================
# 📌 Discriminator (PatchGAN with Instance Normalization)
# ====================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(256),
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
torch.cuda.empty_cache()

if __name__ == "__main__":
    num_epochs = 20

    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, (zebra, horse) in progress_bar:
            zebra, horse = zebra.to(device), horse.to(device)
            batch_size = zebra.size(0)

            # Discriminator update
            fake_horse = G_h(zebra)
            fake_zebra = G_z(horse)
            D_H_real = D_h(horse)
            D_H_fake = D_h(fake_horse.detach())
            loss_D_B = criterion_GAN(D_H_real, torch.ones_like(D_H_real)) + criterion_GAN(D_H_fake, torch.zeros_like(D_H_fake))

            D_Z_real = D_z(zebra)
            D_Z_fake = D_z(fake_zebra.detach())
            loss_D_A = criterion_GAN(D_Z_real, torch.ones_like(D_Z_real)) + criterion_GAN(D_Z_fake, torch.zeros_like(D_Z_fake))
            
            D_loss = (loss_D_A + loss_D_B) / 2

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Generator update
            D_H_fake = D_h(fake_horse)
            D_Z_fake = D_z(fake_zebra)
            loss_G_H = criterion_GAN(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = criterion_GAN(D_Z_fake, torch.ones_like(D_Z_fake))

            loss_cycle_zebra = criterion_Cycle(zebra, G_z(fake_horse))
            loss_cycle_horse = criterion_Cycle(horse, G_h(fake_zebra))

            loss_G = loss_G_H + loss_G_Z + 10 * (loss_cycle_zebra + loss_cycle_horse)

            opt_gen.zero_grad()
            g_scaler.scale(loss_G).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss G: {loss_G.item():.4f} Loss D: {(loss_D_A.item() + loss_D_B.item()):.4f}")

        # 🎯 Display & Save results every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_image(fake_horse[:8], f"./output/lego_epoch_{epoch+1}.png", normalize=True)
            save_image(fake_zebra[:8], f"./output/reconstructed_epoch_{epoch+1}.png", normalize=True)

    #torch.save(G_h.state_dict(), "generator_face_to_lego.pth")
    #torch.save(G_z.state_dict(), "generator_lego_to_face.pth")
    print("Training complete! Models saved.")
