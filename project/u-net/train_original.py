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

import torchvision.transforms as T

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
    

data_folder = "data"
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/unet-voc.pt"
saving_interval = 5
epoch_number = 30
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
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3)
])

target_transform = MaskTransform()

dataset = datasets.VOCSegmentation(
    data_folder,
    year="2007",
    download=False,
    image_set="train",
    transform=transform,
    target_transform=target_transform,
)


def train():
    cell_dataset = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=shuffle_data_loader)

    print("Defining model...")
    model = UNet(dimensions=14)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    # criterion = nn.CrossEntropyLoss()
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    loss_history = []

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
            loss = 0.7 * loss_ce + 0.3 * loss_dice
            loss.backward()
            optimizer.step()
            

            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"  Batch {i:3d} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(cell_dataset)
        loss_history.append(avg_loss)
        print(f"✅ Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save model at intervals
        if (epoch + 1) % saving_interval == 0:
            torch.save(model.state_dict(), f"{model_folder}/unet-epoch{epoch+1}.pt")


    # Final model save
    torch.save(model.state_dict(), model_path)

    # Plot loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(f"{model_folder}/loss_curve.png")
    print("📉 Loss curve saved as loss_curve.png")

    return



if __name__ == "__main__":
    train()
