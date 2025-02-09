import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import gc  # Garbage collection to free memory
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
# 📌 Data Transformations (Reduced resolution to 64x64)
# ====================
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Reduced resolution to 64x64
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Load Dataset
#dataset = FaceToLegoDataset("./lego_ref_images/lg_cropped", "./lego_ref_images/og_cropped", transform=transform)
dataset = FaceToLegoDataset("./augmented/aug_lg", "./augmented/aug_og", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Reduced batch size to 4

# ====================
# 📌 Residual Block with Instance Normalization (Reduced channels)
# ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels)  # Use InstanceNorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)  # Use InstanceNorm
        
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Add the residual connection
        return self.relu(x)

# ====================
# 📌 Generator (Simplified model with reduced channels and layers)
# ====================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),  # Reduced channels
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        # Ensure the output size is the same as input (64x64)
        x = nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return x

# ====================
# 📌 Discriminator (Simplified model with reduced channels)
# ====================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # Reduced channels
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
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
# 📌 Training Loop with Gradient Accumulation
# ====================
if __name__ == "__main__":
    num_epochs = 50
    accumulation_steps = 4  # Accumulate gradients over 4 mini-batches

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
            if (i + 1) % accumulation_steps == 0:  # Update every accumulation_steps
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
            if (i + 1) % accumulation_steps == 0:  # Update every accumulation_steps
                g_scaler.step(opt_gen)
                g_scaler.update()

            # Free memory
            #del fake_horse, fake_zebra, D_H_real, D_H_fake, D_Z_real, D_Z_fake
            gc.collect()

            progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss G: {loss_G.item():.4f} Loss D: {(loss_D_A.item() + loss_D_B.item()):.4f}")

        # 🎯 Display & Save results every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_image(fake_horse[:16], f"./output/lego_epoch_{epoch+1}.png", normalize=True)
            save_image(fake_zebra[:16], f"./output/reconstructed_epoch_{epoch+1}.png", normalize=True)
        
    torch.save(G_h.state_dict(), "generator_face_to_lego.pth")
    torch.save(G_z.state_dict(), "generator_lego_to_face.pth")

    print("Training complete!")
