import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# ====================
# 📌 Check GPU
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ====================
# 📌 Create Directory for Saved Images
# ====================
os.makedirs("generated_images", exist_ok=True)

# ====================
# 📌 Dataset Preparation
# ====================
class FaceToLegoDataset(Dataset):
    def __init__(self, og_folder, lg_folder, transform=None):
        self.og_folder = og_folder
        self.lg_folder = lg_folder
        self.transform = transform
        self.og_images = sorted(os.listdir(og_folder))  # List of original images
        self.lg_images = sorted(os.listdir(lg_folder))  # List of LEGO images

    def __len__(self):
        return max(len(self.og_images), len(self.lg_images))

    def __getitem__(self, idx):
        # Randomly select one image from each domain
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
    transforms.Resize((128, 128)),  # Change from 256x256 to 128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize between -1 and 1
])

# Load Dataset
dataset = FaceToLegoDataset("./dataset/human", "./dataset/lego", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ====================
# 📌 Generator Model
# ====================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),  # Increased filter size
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)


# ====================
# 📌 Discriminator Model
# ====================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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


# ====================
# 📌 Initialize Models & Apply Weights
# ====================
G_h = Generator().to(device)
G_z = Generator().to(device)
D_h = Discriminator().to(device)
D_z = Discriminator().to(device)

# Loss and Optimizers
criterion_GAN = nn.BCELoss()
criterion_Cycle = nn.MSELoss()

opt_disc = optim.Adam(list(D_h.parameters()) + list(D_z.parameters()), lr=0.00002, betas=(0.5, 0.999))
opt_gen = optim.Adam(list(G_h.parameters()) + list(G_z.parameters()), lr=0.00002, betas=(0.5, 0.999))

# Mixed precision training
scaler_G = torch.cuda.amp.GradScaler()
scaler_D = torch.cuda.amp.GradScaler()

def denorm(tensor):
    """Denormalize the images (reverse the normalization)"""
    return tensor * 0.5 + 0.5

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

    plt.show()

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
            'G_h_state_dict': G_h.state_dict(),
            'G_z_state_dict': G_z.state_dict(),
            'D_h_state_dict': D_h.state_dict(),
            'D_z_state_dict': D_z.state_dict(),
            'opt_gen_state_dict': opt_gen.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict()
        }
        torch.save(checkpoint, f"{checkpoint_path}/best_checkpoint.pth")
        print(f"Best checkpoint saved at epoch {epoch}.")


# ====================
# 📌 Training Loop with Image Saving Every 10 Epochs

num_epochs = 10

def lambda_rule(epoch):
    decay_start_epoch = num_epochs // 2  # Start decaying at halfway point
    return 1.0 - max(0, epoch - decay_start_epoch) / (num_epochs - decay_start_epoch)

scheduler_G = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda_rule)
scheduler_D = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda_rule)
# ====================

for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for i, (real_face, real_lego) in progress_bar:
        real_face, real_lego = real_face.to(device), real_lego.to(device)

        # Discriminator update
        fake_lego = G_h(real_face)  # Generate LEGO version from original face
        fake_face = G_z(real_lego)  # For cycle consistency, generate fake face from LEGO version

        loss_D_h = criterion_GAN(D_h(real_lego), torch.ones_like(D_h(real_lego))) + criterion_GAN(D_h(fake_lego), torch.zeros_like(D_h(fake_lego)))
        loss_D_z = criterion_GAN(D_z(real_face), torch.ones_like(D_z(real_face))) + criterion_GAN(D_z(fake_face), torch.zeros_like(D_z(fake_face)))

        D_loss = (loss_D_h + loss_D_z) / 2
        opt_disc.zero_grad()
        scaler_D.scale(D_loss).backward()
        scaler_D.step(opt_disc)
        scaler_D.update()

        # Generator update
        loss_G = criterion_GAN(D_h(G_h(real_face)), torch.ones_like(D_h(real_face))) + criterion_GAN(D_z(G_z(real_lego)), torch.ones_like(D_z(real_lego)))
        loss_G += 10 * (criterion_Cycle(real_face, G_z(G_h(real_face))) + criterion_Cycle(real_lego, G_h(G_z(real_lego))))

        opt_gen.zero_grad()
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_gen)
        scaler_G.update()

        progress_bar.set_postfix(D_loss=D_loss.item(), G_loss=loss_G.item())

    # Save images every 10 epochs
    #if (epoch + 1) % 10 == 0:
    #    with torch.no_grad():
    #        fake_lego = G_h(real_face)  # Generate the LEGO image from the real face
    #        fake_face = G_z(real_lego)  # Cycle consistency part
##
  ##          save_image(fake_lego * 0.5 + 0.5, f"./output/fake_lego_epoch_{epoch+1}.png")
    #        save_image(fake_face * 0.5 + 0.5, f"./output/fake_face_epoch_{epoch+1}.png")
#
 #           # Show original face and generated LEGO
  #          show_images(real_face, fake_lego, epoch)
        scheduler_G.step()
        scheduler_D.step()
    save_best_checkpoint(epoch + 1, loss_G, D_loss)

print("Training complete! Images saved in 'output/' folder.")













#if (epoch + 1) % 5 == 0:
        #with torch.no_grad():
            #fake_lego = G_h(real_face)  # Generate the LEGO image from the real face
            #fake_face = G_z(real_lego)  # Cycle consistency part

            #save_image(fake_lego * 0.5 + 0.5, f"./output/fake_lego_epoch_{epoch+1}.png")
            #save_image(fake_face * 0.5 + 0.5, f"./output/fake_face_epoch_{epoch+1}.png")

        #show_images(real_face, fake_lego, epoch)



checkpoint = torch.load("./checkpoints/train/best_checkpoint.pth", map_location=device)
print(checkpoint['epoch'])
G_h.load_state_dict(checkpoint['G_h_state_dict'])
G_h.eval()

# Select 10 images from the human dataset
image_filenames = os.listdir("./test_op_images")

# Load and process images
original_images = []
converted_images = []

for img_name in image_filenames:
    img = Image.open(os.path.join("./test_op_images", img_name)).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # Apply transformation and move to GPU
    with torch.no_grad():
        converted_img = G_h(img).cpu().squeeze(0)  # Generate LEGO-style image

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
#plt.savefig("./output/generated_grid_try3.png")
plt.show()