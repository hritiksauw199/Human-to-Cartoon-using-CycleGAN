
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
from torch.optim.lr_scheduler import LambdaLR


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
    

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = FaceToLegoDataset("./dataset/human_no_border", "./dataset/cartoons_without_glasses", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),  # c7s1-64
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # d128
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # d256
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(256),  # R256
            ResidualBlock(256),  # R256
            ResidualBlock(256),  # R256
            ResidualBlock(256)  # R256
        )

        # Upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # u128
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # u64
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),  # c7s1-3
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x
    
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # c64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (input: 3 channels -> 64 filters)
            nn.LeakyReLU(0.2),  # LeakyReLU activation with 0.2 negative slope

            # d128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (64 -> 128 filters)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # d256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (128 -> 256 filters)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # d512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (256 -> 512 filters)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # Final output (single channel: real or fake)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # (512 -> 1 output channel)
            nn.Sigmoid()  # Sigmoid activation to output probabilities
        )

    def forward(self, x):
        return self.model(x)
    

G_lego = Generator().to(device)
G_face = Generator().to(device)
D_lego = PatchGANDiscriminator().to(device)
D_face = PatchGANDiscriminator().to(device)


criterion_GAN = nn.MSELoss()
criterion_Cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

opt_disc = optim.Adam(list(D_lego.parameters()) + list(D_face.parameters()), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)
opt_gen = optim.Adam(list(G_lego.parameters()) + list(G_face.parameters()), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-4)


scaler_G = torch.cuda.amp.GradScaler()
scaler_D = torch.cuda.amp.GradScaler()

def linear_lr_lambda(epoch, total_epochs, start_decay_epoch):
    """ Linearly decays LR after 'start_decay_epoch' until total_epochs. """
    if epoch < start_decay_epoch:
        return 1.0  # Keep learning rate constant
    else:
        return 1.0 - (epoch - start_decay_epoch) / (total_epochs - start_decay_epoch)
    
num_epochs = 30
start_decay_epoch = 15
    
scheduler_G = LambdaLR(opt_gen, lr_lambda=lambda epoch: linear_lr_lambda(epoch, num_epochs, start_decay_epoch))
scheduler_D = LambdaLR(opt_disc, lr_lambda=lambda epoch: linear_lr_lambda(epoch, num_epochs, start_decay_epoch))
    
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

D_lego_loss = []
D_face_loss = []
G_loss_data = []

for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for i, (real_face, real_lego) in progress_bar:
        real_face, real_lego = real_face.to(device), real_lego.to(device)

        # Discriminator update
        fake_face = G_face(real_lego)
        D_face_real = D_face(real_face)
        D_face_fake = D_face(fake_face)
        loss_D_face = criterion_GAN(D_face_real, torch.ones_like(D_face_real)) + \
                    criterion_GAN(D_face_fake, torch.zeros_like(D_face_fake))
            

        fake_lego = G_lego(real_face)  
        D_lego_real = D_lego(real_lego)
        D_lego_fake = D_lego(fake_lego)
        loss_D_lego = criterion_GAN(D_lego_real, torch.ones_like(D_lego_real)) + \
                    criterion_GAN(D_lego_fake, torch.zeros_like(D_lego_fake))

        D_loss = (loss_D_lego + loss_D_face ) / 2


        opt_disc.zero_grad()
        scaler_D.scale(D_loss).backward()
        scaler_D.step(opt_disc)
        scaler_D.update()

        id_Y = G_lego(real_lego)
        id_X = G_face(real_face)
        id_Y_loss = criterion_identity(real_lego, id_Y)
        id_X_loss = criterion_identity(real_face, id_X)

        # Generator update
        G_loss = criterion_GAN(D_lego(G_lego(real_face)), torch.ones_like(D_lego(real_face))) \
            + criterion_GAN(D_face(G_face(real_lego)), torch.ones_like(D_face(real_lego)))
        G_loss += 10 * (criterion_Cycle(real_face, G_face(G_lego(real_face))) + criterion_Cycle(real_lego, G_lego(G_face(real_lego)))) \
         + (id_Y_loss + id_X_loss) * 0.2



        opt_gen.zero_grad()
        scaler_G.scale(G_loss).backward()
        scaler_G.step(opt_gen)
        scaler_G.update()


        progress_bar.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

    D_lego_loss.append(loss_D_lego.item())
    D_face_loss.append(loss_D_face.item())
    G_loss_data.append(G_loss.item())
    
    scheduler_G.step()
    scheduler_D.step()

    save_best_checkpoint(epoch + 1, G_loss, D_loss)


def denorm(tensor):
    """Denormalize the images (reverse the normalization)"""
    return tensor * 0.5 + 0.5



checkpoint = torch.load("./checkpoints/train/best_checkpoint.pth", map_location=device)
print(checkpoint['epoch'])
G_lego.load_state_dict(checkpoint['G_h_state_dict'])
G_lego.eval()

# Select 10 images from the human dataset
image_filenames = os.listdir("./test_op_images/human_test")

# Load and process images
original_images = []
converted_images = []

for img_name in image_filenames:
    img = Image.open(os.path.join("./test_op_images/human_test", img_name)).convert("RGB")
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
plt.savefig("./output/demo_30_lr.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), D_lego_loss, label="Discriminator  Cartoon Loss", color='r')
plt.plot(range(num_epochs), D_face_loss, label="Discriminator  Face Loss", color='g')
plt.plot(range(num_epochs), G_loss_data, label="Generator Loss (G_loss)", color='b')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Discriminator and Generator Losses Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('demo_loss_30_lr.png')
plt.show()


print("Done")