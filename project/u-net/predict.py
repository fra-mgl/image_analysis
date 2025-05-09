import numpy as np
from PIL import Image
import torch

from unet import UNet
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Define VOC colormap and class names (14 classes)
VOC_COLORMAP = [
    (0, 0, 0),         # 0 background
    (128, 0, 0),       # 1
    (0, 128, 0),       # 2
    (128, 128, 0),     # 3
    (0, 0, 128),       # 4
    (128, 0, 128),     # 5
    (0, 128, 128),     # 6
    (128, 128, 128),   # 7
    (64, 0, 0),        # 8
    (192, 0, 0),       # 9
    (64, 128, 0),      # 10
    (192, 128, 0),     # 11
    (64, 0, 128),      # 12
    (192, 0, 128),     # 13
]

VOC_CLASSES = [
    "background", "Amandina", "Arabia", "Comtesse", "Creme brulee",
    "Jelly Black", "Jelly Milk", "Jelly White", "Noblesse", "Noir authentique",
    "Passion au lait", "Stracciatella", "Tentation Noir", "Triangolo"
]

# Resize + preserve integer mask labels
class MaskTransform:
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, mask):
        mask = mask.resize(self.size, resample=Image.NEAREST)
        return torch.as_tensor(np.array(mask), dtype=torch.long)

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
data_folder = "data"
model_path = "model/unet-voc.pt"
shuffle_data_loader = False

# Transforms
transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
target_transform = MaskTransform(size=(512, 512))

# Dataset
dataset = datasets.VOCSegmentation(
    data_folder,
    year="2007",
    download=False,
    image_set="train",
    transform=transform,
    target_transform=target_transform,
)

# Prediction and visualization
def predict():
    model = UNet(dimensions=14)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    cell_dataset = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle_data_loader)

    for i, (input, _) in enumerate(cell_dataset):
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

        if i >= 10:
            break

if __name__ == "__main__":
    predict()

