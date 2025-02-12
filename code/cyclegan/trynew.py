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



class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_downsampling: bool = True,
        add_activation: bool = True,
        **kwargs
    ):
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
    def __init__(
        self, img_channels: int, num_features: int = 64, num_residuals: int = 9
    ):
        
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.ReLU(inplace=True),
        )

        self.downsampling_layers = nn.ModuleList(
            [
                ConvolutionalBlock(
                    num_features, 
                    num_features * 2,
                    is_downsampling=True, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1,
                ),
                ConvolutionalBlock(
                    num_features * 2,
                    num_features * 4,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.residual_layers = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.upsampling_layers = nn.ModuleList(
            [
                ConvolutionalBlock(
                    num_features * 4,
                    num_features * 2,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvolutionalBlock(
                    num_features * 2,
                    num_features * 1,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last_layer = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.downsampling_layers:
            x = layer(x)
        x = self.residual_layers(x)
        for layer in self.upsampling_layers:
            x = layer(x)
        return torch.tanh(self.last_layer(x))


class ConvInstanceNormLeakyReLUBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        Class object initialization for Convolution-InstanceNorm-LeakyReLU layer

        We use leaky ReLUs with a slope of 0.2.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    


# ====================
# 📌 Discriminator (PatchGAN)
# ====================
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        """
        Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with 
        k filters and stride 2. Discriminator architecture is: C64-C128-C256-C512. 
        
        After the last layer, we apply a convolution to produce a 1-dimensional 
        output. 
        
        We use leaky ReLUs with a slope of 0.2.
        """
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                ConvInstanceNormLeakyReLUBlock(
                    in_channels, 
                    feature, 
                    stride=1 if feature == features[-1] else 2,
                )
            )
            in_channels = feature

        # After the last layer, we apply a convolution to produce a 1-dimensional output 
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)

        # feed the model output into a sigmoid function to make a 1/0 label
        return torch.sigmoid(self.model(x))
    


# ====================
# 📌 Initialize Models
# ====================
G_h = Generator(img_channels=3, num_residuals=5).to(device)  # Real Face → LEGO
G_z = Generator(img_channels=3, num_residuals=5).to(device)  # LEGO → Real Face
D_h = Discriminator(in_channels=3).to(device)  # Discriminator for Real Faces
D_z = Discriminator(in_channels=3).to(device)  # Discriminator for LEGO Faces

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
if __name__ == "__main__":
    num_epochs = 100

    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, (zebra, horse) in progress_bar:
            zebra, horse = zebra.to(device), horse.to(device)
            batch_size = zebra.size(0)

            # Discr
            fake_horse = G_h(zebra)
            fake_zebra = G_z(horse)  # A → B (Real Face → LEGO)
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
        if (epoch + 1) % 20 == 0:
            save_image(fake_horse[:8], f"./output/lego_epoch_{epoch+1}.png", normalize=True)  # LEGO generated from Real Faces
            save_image(fake_zebra[:8], f"./output/reconstructed_epoch_{epoch+1}.png", normalize=True)  # Reconstructed Real Faces from LEGO

    torch.save(G_h.state_dict(), "generator_face_to_lego_new.pth")
    torch.save(G_z.state_dict(), "generator_lego_to_face_new.pth")
    print("Training complete! Models saved.")
