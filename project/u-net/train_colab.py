import os
import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms, utils, datasets
import numpy as np
from PIL import Image
import cv2
from torchvision.transforms import functional as F
# 
import matplotlib.pyplot as plt
from unet.unet import UNet
from data.frequencies import compute_class_weights

import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

class TrainingParams():
    def __init__(self, data_folder, model_folder,  epoch_number, saving_interval, bilateral_parameters, shuffle = True):
        self.data_folder = data_folder
        self.model_folder = Path(model_folder)
        self.model_folder.mkdir(exist_ok=True)
        self.model_path = model_folder + "unet_voc"
        self.saving_interval = saving_interval
        self.epoch_number = epoch_number
        self.bilateral_parameters = bilateral_parameters
        self.shuffle_data_loader = shuffle
        
  
class BilateralFilter():
    def __init__(self,params):
        self.d = params.bilateral_parameters[0]
        self.sigma_color = params.bilateral_parameters[1]
        self.sigma_space = params.bilateral_parameters[2]

    def __call__(self, img):
        # Convert PIL Image to NumPy array
        img_np = np.array(img)

        # Apply bilateral filter (for 3-channel color images)
        if img_np.ndim == 3:
            filtered = cv2.bilateralFilter(img_np, self.d, self.sigma_color, self.sigma_space)
        else:
            filtered = img_np  # skip for grayscale

        # Convert back to PIL Image
        return Image.fromarray(filtered)

class ResizeLongestSide:
    def __init__(self, longest=512):
        self.longest = longest

    def __call__(self, img):
        w, h = img.size
        scale = self.longest / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), resample=Image.BILINEAR)
    
class MaskTransform:
    def __init__(self):
        self.resize = ResizeLongestSide(512)

    def __call__(self, mask):
        mask = self.resize(mask)
        return torch.as_tensor(np.array(mask), dtype=torch.long)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        logits = torch.softmax(logits, dim=1)

        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(logits * targets_one_hot, dims)
        cardinality = torch.sum(logits + targets_one_hot, dims)

        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice_score.mean()

def validate(model, val_loader, ce_loss, dice_loss, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss_ce = ce_loss(output, target)
            loss_dice = dice_loss(output, target)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            val_loss += loss.item()

    return val_loss / len(val_loader)

def train(params):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #1. Define transform
    print("Defining transform...")
    # transform = transforms.Compose([
    #     ResizeLongestSide(512),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    #     transforms.GaussianBlur(kernel_size=5, sigma=(1, 2)),
    #     transforms.ToTensor(),
    #     transforms.RandomErasing(p=0.3)
    # ])
    transform = transforms.Compose([
    ResizeLongestSide(512),
    BilateralFilter(params),  
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
    ResizeLongestSide(512),   # Resize with aspect ratio preservation
    transforms.ToTensor()     # Convert to tensor
    ])

    target_transform = MaskTransform()

    #2. Load dataset
    print("Loading dataset...")
    train_dataset = datasets.VOCSegmentation(
        params.data_folder,
        year="2007",
        download=False,
        image_set="train",
        transform=transform,
        target_transform=target_transform,
    )
    val_dataset = datasets.VOCSegmentation(
        params.data_folder,
        year="2007",
        download=False,
        image_set="val",
        transform=val_transform,
        target_transform=target_transform,
        )

    weights = torch.tensor(compute_class_weights(params.data_folder), dtype=torch.float32).to(device)

    print("Class Weights:", weights)
    cell_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle= params.shuffle_data_loader)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    #3. Define model
    print("Defining model...")
    model = UNet(dimensions=14)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    
    ce_loss = nn.CrossEntropyLoss(weight = weights) #
    dice_loss = DiceLoss()

    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_model_path = f"{params.model_path}_best.pt"

    for epoch in range(params.epoch_number):
        print(f"\n🔁 Epoch {epoch+1}/{params.epoch_number}")
        epoch_loss = 0
        model.train()

        for i, batch in enumerate(cell_dataset):
            input, target = batch
            input = input.to(device)
            target = target.to(device)

            # Skip small batches
            if input.shape[0] < 2:
                continue

            optimizer.zero_grad()
            output = model(input)

            #loss = criterion(output, target)
            loss_ce = ce_loss(output, target)
            loss_dice = dice_loss(output, target)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"  Batch {i:3d} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(cell_dataset)
        val_avg_loss = validate(model, val_loader, ce_loss, dice_loss, device)
        loss_history.append(avg_loss)
        val_loss_history.append(val_avg_loss)
        print(f"✅ Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_avg_loss:.4f}")

        # Save model at intervals
        if (epoch + 1) % params.saving_interval == 0:
            torch.save(model.state_dict(), f"{params.model_path}{epoch+1}.pt")
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Best model saved with val loss {val_avg_loss:.4f}")


    # Final model save
    torch.save(model.state_dict(),params.model_path)

    # Plot loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train Loss", "Validation Loss"])
    plt.title("Training Loss Curves")
    plt.savefig(f"{params.model_folder}/loss_curve.png")
    print("📉 Loss curve saved as loss_curve.png")

    return



if __name__ == "__main__":
    params = TrainingParams(
        data_folder="data/",
        model_folder="model/",
        epoch_number=30,
        saving_interval=10,
        bilateral_parameters=(5, 75, 75),  # d, sigma_color, sigma_space
        shuffle=True
    )
    train(params)