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


# ====================
# 📌 Training Loop with Image Saving Every 10 Epochs
# ====================

num_epochs = 50

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
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake_lego = G_h(real_face)  # Generate the LEGO image from the real face
            fake_face = G_z(real_lego)  # Cycle consistency part

            save_image(fake_lego * 0.5 + 0.5, f"./output/fake_lego_epoch_{epoch+1}.png")
            save_image(fake_face * 0.5 + 0.5, f"./output/fake_face_epoch_{epoch+1}.png")

            # Show original face and generated LEGO
            show_images(real_face, fake_lego, epoch)

print("Training complete! Images saved in 'output/' folder.")













#if (epoch + 1) % 5 == 0:
        #with torch.no_grad():
            #fake_lego = G_h(real_face)  # Generate the LEGO image from the real face
            #fake_face = G_z(real_lego)  # Cycle consistency part

            #save_image(fake_lego * 0.5 + 0.5, f"./output/fake_lego_epoch_{epoch+1}.png")
            #save_image(fake_face * 0.5 + 0.5, f"./output/fake_face_epoch_{epoch+1}.png")

        #show_images(real_face, fake_lego, epoch)



G_h = Generator().to(device)
checkpoint = torch.load("./checkpoints/train/best_checkpoint.pth", map_location=device)
G_h.load_state_dict(checkpoint['G_h_state_dict'])
G_h.eval()  # Set to evaluation mode

# ====================
# 📌 Load 10 Random Human Images
# ====================
human_folder = "./dataset/human"
human_images = random.sample(os.listdir(human_folder), 10)  # Pick 10 random images

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Same normalization as training
])

real_faces = []
for img_name in human_images:
    img = Image.open(os.path.join(human_folder, img_name)).convert("RGB")
    img = transform(img)
    real_faces.append(img)

real_faces = torch.stack(real_faces).to(device)  # Convert list to tensor batch

# ====================
# 📌 Generate LEGO Images
# ====================
with torch.no_grad():  # No need to calculate gradients
    fake_legos = G_h(real_faces)

# ====================
# 📌 Denormalize Function
# ====================
def denorm(tensor):
    return tensor * 0.5 + 0.5  # Reverse normalization

# ====================
# 📌 Show Images in Grid (5x2)
# ====================
real_faces = denorm(real_faces).cpu()
fake_legos = denorm(fake_legos).cpu()

fig, axes = plt.subplots(5, 2, figsize=(10, 15))  # 5 rows, 2 columns

for i in range(5):
    # Real Face
    axes[i, 0].imshow(real_faces[i].permute(1, 2, 0))  # Convert to (H, W, C)
    #axes[i, 0].set_title("Real Face")
    axes[i, 0].axis("off")

    # Generated LEGO
    axes[i, 1].imshow(fake_legos[i].permute(1, 2, 0))
    #axes[i, 1].set_title("Generated LEGO")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig("output_grid.png")  # Save output
plt.show()  # Show grid