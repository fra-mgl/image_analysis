import os
import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms, utils, datasets
import numpy as np
from PIL import Image
# 
import matplotlib.pyplot as plt
from unet.unet import UNet
from data.frequencies import compute_class_weights



import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

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

def validate(model, val_loader, ce_loss, dice_loss):
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

    

data_folder = "data"
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/unet-voc.pt"
saving_interval = 5
epoch_number = 80
shuffle_data_loader = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
# dataset = datasets.VOCSegmentation(
#     data_folder,
#     year="2007",
#     download=False,
#     image_set="train",
#     transform=transform,
#     target_transform=transform,
# )

# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor()
# ])
  #1. Define transform
print("Defining transform...")
transform = transforms.Compose([
    ResizeLongestSide(512),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.85, 1.85)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2)
])


val_transform = transforms.Compose([
    ResizeLongestSide(512),   # Resize with aspect ratio preservation
    transforms.ToTensor()     # Convert to tensor
])


target_transform = MaskTransform()

train_dataset = datasets.VOCSegmentation(
    data_folder,
    year="2007",
    download=False,
    image_set="train",
    transform=transform,
    target_transform=target_transform,
)

val_dataset = datasets.VOCSegmentation(
    data_folder,
    year="2007",
    download=False,
    image_set="val",
    transform=val_transform,
    target_transform=target_transform,
)

weights = torch.tensor(compute_class_weights(), dtype=torch.float32).to(device)


def train():
    cell_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=shuffle_data_loader)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)


    print("Defining model...")
    model = UNet(dimensions=14)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    # criterion = nn.CrossEntropyLoss()
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_loss = DiceLoss()

    loss_history = []
    val_loss_history = []

    for epoch in range(epoch_number):
        print(f"\n🔁 Epoch {epoch+1}/{epoch_number}")
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
            loss = 0.5 * loss_ce + 0.5* loss_dice #before 0.7 and 0.3
            loss.backward()
            optimizer.step()
            

            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"  Batch {i:3d} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(cell_dataset)
        val_avg_loss = validate(model, val_loader, ce_loss, dice_loss)
        loss_history.append(avg_loss)
        val_loss_history.append(val_avg_loss)
        print(f"✅ Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_avg_loss:.4f}")

        # Save model at intervals
        if (epoch + 1) % saving_interval == 0:
            torch.save(model.state_dict(), f"{model_folder}/unet-epoch{epoch+1}.pt")


    # Final model save
    torch.save(model.state_dict(), model_path)

    # Plot loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.plot(val_loss_history)
    plt.legend(["Train Loss", "Validation Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.savefig(f"{model_folder}/loss_curve.png")
    print("📉 Loss curve saved as loss_curve.png")

    return



if __name__ == "__main__":
    train()
