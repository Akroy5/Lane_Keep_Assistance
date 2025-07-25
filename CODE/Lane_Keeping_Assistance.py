# Lane Keeping Assistance (LKA) System
# Project: IoT-Fused CNN for Lane-Keeping from Scratch
# CODE/model.py

import torch
import torch.nn as nn

class LKADeepIoTFusion(nn.Module):
    """
    CNN model for fusing RGB + Radar + LiDAR + Ultrasonic (total 6 channels)
    Output: Lane segmentation (binary mask) or regression map
    """
    def __init__(self):
        super(LKADeepIoTFusion, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 1, 1)  # Single channel output (logits for BCE)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

# CODE/dataset.py
# ----------------
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch

class LKADataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = sorted(os.listdir(os.path.join(data_dir, 'images')))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]
        rgb = Image.open(os.path.join(self.data_dir, 'images', fname)).convert('RGB')
        radar = Image.open(os.path.join(self.data_dir, 'radar', fname)).convert('L')
        lidar = Image.open(os.path.join(self.data_dir, 'lidar', fname)).convert('L')
        ultra = Image.open(os.path.join(self.data_dir, 'ultrasonic', fname)).convert('L')
        mask = Image.open(os.path.join(self.data_dir, 'masks', fname)).convert('L')

        if self.transform:
            rgb = self.transform(rgb)
            radar = self.transform(radar)
            lidar = self.transform(lidar)
            ultra = self.transform(ultra)
            mask = self.transform(mask)

        fused = torch.cat([rgb, radar, lidar, ultra], dim=0)  # 6 channels total
        return fused, mask

# CODE/train.py
# --------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LKADeepIoTFusion
from dataset import LKADataset
from torchvision import transforms

# Config
batch_size = 16
lr = 0.001
epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_dataset = LKADataset("DATA/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = LKADeepIoTFusion().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train loop
def train():
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}")

if __name__ == '__main__':
    train()

# CODE/utils.py

import torch

def compute_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0, 1).sum()
    return intersection / (union + 1e-6)

# run_training.sh
# ----------------
# chmod +x CODE/run_training.sh
# bash CODE/run_training.sh
python CODE/train.py
