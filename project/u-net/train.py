import os
import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms, utils, datasets

from unet import UNet

# data_folder = "data"
# model_folder = Path("model")
# model_folder.mkdir(exist_ok=True)
# model_path = "model/unet-voc.pt"
# saving_interval = 10
# epoch_number = 100
# shuffle_data_loader = False
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Grayscale()])
# dataset = datasets.VOCSegmentation(
#     data_folder,
#     year="2007",
#     download=True,
#     image_set="train",
#     transform=transform,
#     target_transform=transform,
# )
# Dataset preparation
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

from torchvision import transforms
from torch.utils.data import DataLoader, random_split

class TrainingParams():
    def __init__(self, folder_path, model_path, epoch_number, cell_dataset, saving_interval, output_dir, train_size, val_size):
        self.folder_path = folder_path,
        self.model_path = model_path
        self.epoch_number = epoch_number
        self.cell_dataset = cell_dataset
        self.saving_interval = saving_interval
        self.output_dir = output_dir
        self.train_size = train_size
        self.val_size = val_size


class ChocolateDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = str(row['id'])
        image_name = f"L{img_id}.JPG"
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row[1:].values, dtype=torch.float32)  # Chocolate counts
        return image, label
    

# NOW dataset is function parameter
def train(params, full_dataset):
    #DATASET PREPARATION
    # 1. Define your transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 2. Load the full dataset
    # full_dataset = ChocolateDataset(
    #     csv_file="/home/elisa/image_analysis/project/dataset_project_iapr2025/train.csv",
    #     image_dir=""
    #     mask_dir="",
    #     transform=transform
    #     #target_transform=transforms.Resize((512, 512))
    # )

    # 3. Split into train and val (e.g. 80 train, 10 val)
    train_dataset, val_dataset = random_split(full_dataset, [params.train_size, params.val_size], generator=torch.Generator().manual_seed(42))

    # 4. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
            
    #------------------------------------
    model = UNet(dimensions=22)
    model.to(device)
    if os.path.isfile(params.model_path): # FIXME questo deve essere il path del file del modello?
        model.load_state_dict(torch.load(params.model_path, map_location=torch.device(device)))
    optimizer = optim.RMSprop(
        model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(params.epoch_number):
        print(f"Epoch {epoch}")
        losses = []
        for i, batch in enumerate(params.cell_dataset):
            input, target = batch
            input = input.to(device)
            target = target.type(torch.LongTensor).to(device)
            # HACK to skip the last item that has a batch size of 1, not working with the cross entropy implementation
            if input.shape[0] < 2:
                continue
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target.squeeze())
            # step_loss = loss.item()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # print the average loss for that epoch.
        print(sum(losses) /len(losses))
        if (epoch + 1) % params.saving_interval == 0:
            print("Saving model")

        torch.save(model.state_dict(), params.model_path)
    torch.save(model.state_dict(), params.model_path)
    return


# if __name__ == "__main__":
#     train()
