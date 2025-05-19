import numpy as np
from PIL import Image
import torch

from src.unet.unet.unet import UNet
from torchvision import transforms
from matplotlib.patches import Patch
import os
from torchvision.datasets.vision import VisionDataset

import csv
from skimage import measure
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes
from matplotlib.patches import Patch
from PIL import Image

from enum import Enum

# Define VOC colormap and class names (14 classes)
VOC_COLORMAP = [
    (0, 0, 0),          # 0 background - black
    (230, 25, 75),      # 1 Amandina - strong red
    (60, 180, 75),      # 2 Arabia - green
    (255, 225, 25),     # 3 Comtesse - yellow
    (0, 130, 200),      # 4 Creme brulee - blue
    (245, 130, 48),     # 5 Jelly Black - orange
    (145, 30, 180),     # 6 Jelly Milk - purple
    (128, 128, 128),    # 7 Jelly White - grey
    (70, 240, 240),     # 8 Noblesse - cyan
    (240, 50, 230),     # 9 Noir authentique - pink/magenta
    (210, 245, 60),     # 10 Passion au lait - lime/yellow-green
    (250, 190, 190),    # 11 Stracciatella - light pink
    (0, 128, 128),      # 12 Tentation Noir - teal
    (255, 215, 180),    # 13 Triangolo - peach
]

VOC_CLASSES = [
    "background", "Amandina", "Arabia", "Comtesse", "Creme brulee",
    "Jelly Black", "Jelly Milk", "Jelly White", "Noblesse", "Noir authentique",
    "Passion au lait", "Stracciatella", "Tentation Noir", "Triangolo"
]

CSV_CLASSES = [
    "Jelly White", "Jelly Milk", "Jelly Black", "Amandina", "Creme brulee",
    "Triangolo", "Tentation Noir", "Comtesse", "Noblesse", "Noir authentique",
    "Passion au lait", "Arabia", "Stracciatella"
]
VOCEnum = Enum('VOCEnum', {name: idx for idx, name in enumerate(VOC_CLASSES)})

area_threshold = {
    1: 2224,
    2: 2255,
    3: 2231,
    4: 2445,
    5: 1151,
    6: 1055,
    7: 906,
    8: 1389,
    9: 1936,
    10: 2234,
    11: 1484,
    12: 1735,
    13: 1761
}

area_scaling = 0.93

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
    
from torchvision.datasets.vision import VisionDataset

class VOCInferenceDataset(VisionDataset):
    def __init__(self, root, image_set="test", transforms=None):
        super().__init__(root, transforms=transforms)
        base_dir = os.path.join("src/unet/", root, "VOC2007")
        image_dir = os.path.join(base_dir, "JPEGImages")
        splits_dir = os.path.join(base_dir, "ImageSets", "Segmentation")
        
        with open(os.path.join(splits_dir, image_set + ".txt"), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, f"{x}.jpg") for x in file_names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.images[index]  # return path for reference



# Decode class indices into RGB image
def decode_segmap(mask, colormap=VOC_COLORMAP):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in enumerate(colormap):
        color_mask[mask == label] = color
    return color_mask

# Create legend patches
def create_legend(classes, colormap):
    legend_patches = [
        Patch(color=np.array(color) / 255.0, label=cls)
        for cls, color in zip(classes, colormap)
    ]
    return legend_patches

# Paths
data_folder = "data/VOCdevkit/"
model_path = "src/unet/model/unet_voc_best.pt"
shuffle_data_loader = True

# Transforms
transform = transforms.Compose([
    ResizeLongestSide(512),
    transforms.ToTensor()
])
target_transform = MaskTransform()

# Dataset
dataset = VOCInferenceDataset(
    root="data/VOCdevkit/",
    image_set="test",
    transforms=transform
)


def count_instances(area, cls):
    return np.round(area / (area_scaling*area_threshold[cls]), 0)

def post_processing(img_id, img, csv_writer):
    mask = np.array(img, dtype=np.uint8)
    
    # POST PROCESSING
    binary_img = mask > 0
    # binary_img = opening(binary_img, disk(2))  # TODO slightly better using also this
    binary_img = remove_small_holes(binary_img, 300) 
    binary_img = remove_small_objects(binary_img, 300)

    #  COUNT

    # EXTRACT CC
    # Label connected components
    label_image, label_num = measure.label(binary_img, return_num=True)

    improved_mask = np.zeros_like(binary_img, dtype=int)

    prediction_counts = np.zeros(14, dtype=int)
    for i in range(1, label_num+1):
        region = mask[label_image==i] 
        region_mask = np.where(label_image==i, mask, 0)

        class_bins = np.bincount(region)
        for j, b in enumerate(class_bins):
                if j < 14 and j != 0 and b != 0:
                    inst = count_instances(b, j)
                    prediction_counts[j] += inst
                    if inst > 0:
                        improved_mask[region_mask==j] = j            

    # append to csv
    row = [int(img_id)]
    for label in CSV_CLASSES:
        index = VOCEnum[label].value
        row.append(int(prediction_counts[index]))
    csv_writer.writerow(row)

if __name__ == "__main__":
    model = UNet(dimensions=14)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    cell_dataset = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle_data_loader)


    with open('submission.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['id', 'Jelly White', 'Jelly Milk', 'Jelly Black', 'Amandina', 'Crème brulée', 'Triangolo', 'Tentation noir', 'Comtesse', 'Noblesse', 'Noir authentique', 'Passion au lait', 'Arabia', 'Stracciatella'])

        # Prediction
        for i, (input, path) in enumerate(cell_dataset):
            img_id = os.path.basename(path[0])
            img_id = img_id.split('.')[0][1:]

            with torch.no_grad():
                output = model(input)
            output_array = output.argmax(dim=1).squeeze().numpy().astype(np.uint8)

            # Post processing and count
            post_processing(img_id, output_array, csv_writer=writer)
