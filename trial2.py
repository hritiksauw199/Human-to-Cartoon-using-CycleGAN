import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import os
import gc  # Garbage collection to free memory
from tqdm import tqdm
#import albumentations as A  # For advanced augmentations

# ====================
# 📌 Check GPU
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ====================
# 📌 Dataset Preparation with Advanced Augmentation
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
# 📌 Data Augmentation with Random Cropping and Elastic Distortions
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
dataset = FaceToLegoDataset("./lego_ref_images/lg_cropped", "./lego_ref_images/og_cropped", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Reduced batch size to 4

# ====================
# 📌 Generator with U-Net and Self-Attention
# ====================
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, c, h, w = x.size()

        # Apply convolution to get query, key, and value
        query = self.query_conv(x).view(batch_size, -1, h * w).permute(0, 2, 1)  # [batch_size, seq_len, channels]
        key = self.key_conv(x).view(batch_size, -1, h * w)  # [batch_size, seq_len, channels]
        value = self.value_conv(x).view(batch_size, -1, h * w)  # [batch_size, seq_len, channels]

        # Calculate attention
        attention = torch.matmul(query, key)  # [batch_size, seq_len, seq_len]
        attention = torch.nn.functional.softmax(attention, dim=-1)  # Softmax along seq_len

        # Apply attention to value
        out = torch.matmul(attention, value.permute(0, 2, 1))  # [batch_size, seq_len, channels]
        out = out.permute(0, 2, 1).view(batch_size, c, h, w)  # Reshape back to the original size

        return self.gamma * out + x  # Return the weighted sum with residual connection


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            SelfAttention(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ====================
# 📌 Discriminator with LeakyReLU and Self-Attention
# ====================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            SelfAttention(128),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# ====================
# 📌 WGAN-GP Loss
# ====================
def gradient_penalty(critic, real, fake):
    batch_size, c, h, w = real.size()
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    d_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated, inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated).to(device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    return ((gradient_norm - 1) ** 2).mean()

# ====================
# 📌 Initialize Models
# ====================
G = Generator().to(device)
D = Discriminator().to(device)

# ====================
# 📌 Optimizers
# ====================
opt_disc = optim.RMSprop(D.parameters(), lr=0.00005)
opt_gen = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))

# ====================
# 📌 Training Loop with WGAN-GP and Gradient Penalty
# ====================
if __name__ == "__main__":
    num_epochs = 500
    accumulation_steps = 4  # Accumulate gradients over 4 mini-batches

    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, (real, fake) in progress_bar:
            real, fake = real.to(device), fake.to(device)
            batch_size = real.size(0)

            # Discriminator update
            fake_images = G(real)
            D_real = D(real)
            D_fake = D(fake_images.detach())
            loss_D = -(torch.mean(D_real) - torch.mean(D_fake))  # Wasserstein loss
            loss_gp = gradient_penalty(D, real, fake_images)
            loss_D_total = loss_D + 10 * loss_gp

            opt_disc.zero_grad()
            loss_D_total.backward(retain_graph=True)
            opt_disc.step()

            # Generator update
            D_fake = D(fake_images)
            loss_G = -torch.mean(D_fake)  # Wasserstein loss

            opt_gen.zero_grad()
            loss_G.backward(retain_graph=True)
            opt_gen.step()

            progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss G: {loss_G.item():.4f} Loss D: {loss_D_total.item():.4f}")

        # Save images at intervals
        if (epoch + 1) % 50 == 0:
            save_image(fake_images[:16], f"./output/lego_epoch_{epoch+1}.png", normalize=True)

    torch.save(G.state_dict(), "generator_face_to_lego.pth")
    torch.save(D.state_dict(), "discriminator.pth")
    print("Training complete!")
