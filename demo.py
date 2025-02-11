
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm

# ====================
# 📌 Check GPU
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class FaceToLegoDataset(Dataset):
    def __init__(self, og_folder, lg_folder, transform=None):
        self.og_folder = og_folder
        self.lg_folder = lg_folder
        self.transform = transform
        self.og_images = sorted(os.listdir(og_folder))
        self.lg_images = sorted(os.listdir(lg_folder))

    def __len__(self):
        return max(len(self.og_images), len(self.lg_images))

    def __getitem__(self, idx):
        og_img_name = random.choice(self.og_images)
        lg_img_name = random.choice(self.lg_images)

        og_image = Image.open(os.path.join(self.og_folder, og_img_name)).convert("RGB")
        lg_image = Image.open(os.path.join(self.lg_folder, lg_img_name)).convert("RGB")

        if self.transform:
            og_image = self.transform(og_image)
            lg_image = self.transform(lg_image)

        return og_image, lg_image

# ====================
# 📌 Data Transformations
# ====================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = FaceToLegoDataset("./dataset/human", "./dataset/lego", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ====================
# 📌 Weight Initialization
# ====================
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def denorm(tensor):
    """Denormalize the images (reverse the normalization)"""
    return (tensor + 1) / 2

def show_images(real_face, fake_lego, epoch):
    """Function to show the original face and the generated LEGO image"""
    real_face = denorm(real_face[0]).cpu().numpy().transpose((1, 2, 0))
    fake_lego = denorm(fake_lego[0]).cpu().numpy().transpose((1, 2, 0))

    # Plot the images
    plt.figure(figsize=(10, 5))

    # Original face
    plt.subplot(1, 2, 1)
    plt.imshow(real_face)
    plt.title("Original Face")
    plt.axis('off')

    # Generated LEGO image
    plt.subplot(1, 2, 2)
    plt.imshow(fake_lego)
    plt.title(f"Generated LEGO Image (Epoch {epoch+1})")
    plt.axis('off')

    plt.savefig(f'./output/7_compare{epoch+1}.png')
    #plt.show()

# ====================
# 📌 PatchGAN Discriminator
# ====================
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Increased filter size
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

# ====================
# 📌 Generator
# ====================
class Generator(nn.Module):
    def __init__(self, residual_blocks=4):  
        super(Generator, self).__init__()

        # Initial convolution layer (c7s1-64)
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling layers (d128, d256)
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Residual blocks (R256)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(residual_blocks)]
        )

        # Upsampling layers (u128, u64)
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Final convolution layer (c7s1-3)
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsampling(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        return self.final(x)

# ====================
# 📌 Residual Block
# ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

# ====================
# 📌 Initialize Models & Apply Weights
# ====================
G_lego = Generator().to(device)
G_face = Generator().to(device)
D_lego = PatchGANDiscriminator().to(device)
D_face = PatchGANDiscriminator().to(device)

G_lego.apply(weights_init)
G_face.apply(weights_init)
D_lego.apply(weights_init)
D_face.apply(weights_init)

# Loss and Optimizers
criterion_GAN = nn.MSELoss()
criterion_Cycle = nn.L1Loss()

opt_disc = optim.Adam(list(D_lego.parameters()) + list(D_face.parameters()), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-3)
opt_gen = optim.Adam(list(G_lego.parameters()) + list(G_face.parameters()), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-3)

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()
# ====================
# 📌 Checkpoint Management
# ====================
checkpoint_path = "./checkpoints/train"
os.makedirs(checkpoint_path, exist_ok=True)

best_G_loss = float('inf')
best_D_loss = float('inf')

def save_best_checkpoint(epoch, G_loss, D_loss):
    global best_G_loss, best_D_loss
    if G_loss < best_G_loss:
        best_G_loss = min(G_loss, best_G_loss)
        #best_D_loss = min(D_loss, best_D_loss)
        
        checkpoint = {
            'epoch': epoch,
            'G_h_state_dict': G_lego.state_dict(),
            'G_z_state_dict': G_face.state_dict(),
            'D_h_state_dict': D_lego.state_dict(),
            'D_z_state_dict': D_face.state_dict(),
            'opt_gen_state_dict': opt_gen.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict()
        }
        torch.save(checkpoint, f"{checkpoint_path}/best_checkpoint.pth")
        print(f"Best checkpoint saved at epoch {epoch}.")
# ====================
# 📌 Training Loop
# ====================

num_epochs = 10  # Set your desired number of epochs

def lambda_rule(epoch):
    decay_start_epoch = num_epochs // 2  # Start decaying at halfway point
    return 1.0 - max(0, epoch - decay_start_epoch) / (num_epochs - decay_start_epoch)

scheduler_G = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda_rule)
scheduler_D = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda_rule)

for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for i, (real_face, real_lego) in progress_bar:
        real_face, real_lego = real_face.to(device), real_lego.to(device)

        # =========================
        # 1️⃣ Discriminator Update
        # =========================
        
        fake_face = G_face(real_lego)
        D_face_real = D_face(real_face)
        D_face_fake = D_face(fake_face.detach())
        loss_D_face = criterion_GAN(D_face_real, torch.ones_like(D_face_real)) + \
                    criterion_GAN(D_face_fake, torch.zeros_like(D_face_fake))
            

        fake_lego = G_lego(real_face)  
        D_lego_real = D_lego(real_lego)
        D_lego_fake = D_lego(fake_lego.detach())
        loss_D_lego = criterion_GAN(D_lego_real, torch.ones_like(D_lego_real)) + \
                    criterion_GAN(D_lego_fake, torch.zeros_like(D_lego_fake))
            

        D_loss = (loss_D_lego + loss_D_face ) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        #opt_disc.zero_grad()
        #D_loss.backward()
        #opt_disc.step()

        # =========================
        # 2️⃣ Generator Update
        # =========================
        # identity loss
        
        id_lego = G_lego(real_lego)
        id_face = G_face(real_face)
        id_lego_loss = criterion_Cycle(real_lego, id_lego)
        id_face_loss = criterion_Cycle(real_face, id_face)

        print(torch.min(fake_lego), torch.max(fake_lego))
        
        G_loss = criterion_GAN(D_lego(G_lego(real_face)), torch.ones_like(D_lego(real_face))) \
            + criterion_GAN(D_face(G_face(real_lego)), torch.ones_like(D_face(real_lego)))
        G_loss += 10 * (criterion_Cycle(real_face, G_face(G_lego(real_face))) + criterion_Cycle(real_lego, G_lego(G_face(real_lego)))) \
                + (id_lego_loss + id_face_loss) * 5   


        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward(retain_graph=True)
        g_scaler.step(opt_gen)
        g_scaler.update()
        #opt_gen.zero_grad()
        #G_loss.backward()
        #opt_gen.step()

        progress_bar.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

        scheduler_G.step()
        scheduler_D.step()

    # Save checkpoint and show images after every epoch
    save_best_checkpoint(epoch + 1, G_loss, D_loss)


checkpoint = torch.load("./checkpoints/train/best_checkpoint.pth", map_location=device)
print(checkpoint['epoch'])
G_lego.load_state_dict(checkpoint['G_h_state_dict'])
G_lego.eval()

# Select 10 images from the human dataset
image_filenames = os.listdir("./test_op_images")

# Load and process images
original_images = []
converted_images = []

for img_name in image_filenames:
    img = Image.open(os.path.join("./test_op_images", img_name)).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # Apply transformation and move to GPU
    with torch.no_grad():
        converted_img = G_lego(img).cpu().squeeze(0)  # Generate LEGO-style image

    original_images.append(denorm(img.squeeze(0)).cpu())  # Store original
    converted_images.append(denorm(converted_img))  # Store converted

# Arrange images in a grid (5 rows, 2 images per row)
fig, axes = plt.subplots(5, 4, figsize=(10, 12))  # 5 rows, 4 columns (image-converted pairs)

for i in range(5):
    for j in range(2):  # Two pairs per row
        index = i * 2 + j
        if index < len(original_images):
            # Show original image
            axes[i, j * 2].imshow(original_images[index].permute(1, 2, 0))
            axes[i, j * 2].axis("off")
            
            # Show converted LEGO image next to it
            axes[i, j * 2 + 1].imshow(converted_images[index].permute(1, 2, 0))
            axes[i, j * 2 + 1].axis("off")

plt.tight_layout()
plt.savefig("./output/generated_grid_try3.png")
plt.show()


print("Done")