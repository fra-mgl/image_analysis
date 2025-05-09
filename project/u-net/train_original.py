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


class MaskTransform:
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, mask):
        # Resize mask using NEAREST to preserve class indices
        mask = mask.resize(self.size, resample=Image.NEAREST)
        # Convert to LongTensor without normalizing
        return torch.as_tensor(np.array(mask), dtype=torch.long)


data_folder = "data"
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/unet-voc.pt"
saving_interval = 5
epoch_number = 50
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

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

target_transform = MaskTransform(size=(512, 512))

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

    model = UNet(base_channels = 39, dimensions=14)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

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

            loss = criterion(output, target)
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
