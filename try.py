import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset
class FaceToLegoDataset(Dataset):
    def __init__(self, og_folder, lg_folder, transform=None):
        self.og_folder = og_folder
        self.lg_folder = lg_folder
        self.transform = transform
        self.image_filenames = sorted(os.listdir(og_folder))
    
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

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
#dataset = FaceToLegoDataset("./lego_ref_images/og", "./lego_ref_images/lg", transform=transform)
dataset = FaceToLegoDataset("./augmented/aug_og", "./augmented/aug_lg", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and Optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 70
for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (real_faces, real_legos) in progress_bar:
        real_faces, real_legos = real_faces.to(device), real_legos.to(device)
        batch_size = real_faces.size(0)
        
        # Create real and fake labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train Generator
        optimizer_G.zero_grad()
        fake_legos = generator(real_faces)
        fake_preds = discriminator(fake_legos)
        g_loss = criterion(fake_preds, real_labels)
        g_loss.backward()
        optimizer_G.step()
        
        # Train Discriminator
        optimizer_D.zero_grad()
        real_preds = discriminator(real_legos)
        d_real_loss = criterion(real_preds, real_labels)
        fake_preds = discriminator(fake_legos.detach())
        d_fake_loss = criterion(fake_preds, fake_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")
    
    # Save sample output
    if (epoch + 1) % 10 == 0:
        save_image(make_grid(fake_legos[:16], normalize=True), f"output_epoch_{epoch+1}.png")

# Save models
#torch.save(generator.state_dict(), "generator.pth")
#torch.save(discriminator.state_dict(), "discriminator.pth")
print("Training complete! Models saved.")
