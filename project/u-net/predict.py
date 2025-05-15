import numpy as np
from PIL import Image
import torch

from unet.unet import UNet
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from torchvision.datasets.vision import VisionDataset

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
        base_dir = os.path.join(root, "VOC2007")
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
model_path = "model/Colab1/unet_voc80.pt"
shuffle_data_loader = True

# Transforms
#transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
transform = transforms.Compose([
    ResizeLongestSide(512),
    transforms.ToTensor()
])
target_transform = MaskTransform()

# Dataset
# dataset = datasets.VOCSegmentation(
#     data_folder,
#     year="2007",
#     download=False,
#     image_set="test",
#     transform=transform,
#     target_transform=target_transform,
# )
dataset = VOCInferenceDataset(
    root=data_folder,
    image_set="test",
    transforms=transform
)


# Prediction and visualization
def predict():
    model = UNet(dimensions=14)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    cell_dataset = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle_data_loader)

    #for i, (input, _) in enumerate(cell_dataset):
    for i, (input, path) in enumerate(cell_dataset):

        with torch.no_grad():
            output = model(input)

        # Prepare image and mask
        input_np = (input.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        output_array = output.argmax(dim=1).squeeze().numpy().astype(np.uint8)
        output_rgb = decode_segmap(output_array)

        # Plot input and colored segmentation
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_np)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(output_rgb)
        plt.title("Predicted Segmentation")
        plt.axis("off")

        # Add color legend
        legend_patches = create_legend(VOC_CLASSES, VOC_COLORMAP)
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.tight_layout()
        plt.show()

        if i >= 20:
            break

if __name__ == "__main__":
    predict()
