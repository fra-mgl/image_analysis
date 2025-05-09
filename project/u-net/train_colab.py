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
from unet import UNet

class TrainingParams():
    def __init__(self, data_folder, model_folder,  epoch_number, saving_interval, shuffle = True):
        self.data_folder = data_folder
        self.model_folder = Path(model_folder)
        self.model_folder.mkdir(exist_ok=True)
        self.model_path = model_folder + "unet_voc"
        self.saving_interval = saving_interval
        self.epoch_number = epoch_number
        self.shuffle_data_loader = shuffle
  


class MaskTransform:
    def __init__(self):
        self.resize = transforms.Resize(512, interpolation=Image.NEAREST)
        self.crop = transforms.CenterCrop((1365, 2048))

    def __call__(self, mask):
        mask = self.resize(mask)
        mask = self.crop(mask)
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

 
def train(params):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #1. Define transform
    print("Defining transform...")
    transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop((1365, 2048)),  # fixed aspect-ratio shape
    transforms.ColorJitter(brightness=0.5, contrast=1, saturation=0.5),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(5, 10)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2)
    ])


    target_transform = MaskTransform()

    #2. Load dataset
    print("Loading dataset...")
    dataset = datasets.VOCSegmentation(
        params.data_folder,
        year="2007",
        download=False,
        image_set="train",
        transform=transform,
        target_transform=target_transform,
    )
    cell_dataset = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle= params.shuffle_data_loader)

    #3. Define model
    print("Defining model...")
    model = UNet(dimensions=14)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    #criterion = nn.CrossEntropyLoss()
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()


    loss_history = []

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
        if (epoch + 1) % params.saving_interval == 0:
            torch.save(model.state_dict(), f"{params.model_path}{epoch+1}.pt")


    # Final model save
    torch.save(model.state_dict(),params.model_path)

    # Plot loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(f"{params.model_folder}/loss_curve.png")
    print("📉 Loss curve saved as loss_curve.png")

    return



if __name__ == "__main__":
    params = TrainingParams(
        data_folder="data/",
        model_folder="model/",
        epoch_number=30,
        saving_interval=10,
        shuffle=True
    )
    train(params)