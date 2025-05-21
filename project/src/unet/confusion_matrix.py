"""
Filename: confusion_matrix.py
Description: generate confusion matrix to evaluate model performance
Author: Image-inativi
Date: May 21st 2025
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import transforms, datasets 
from unet.unet import UNet
from PIL import Image
import os

# === VOC Settings ===
VOC_CLASSES = [
    "background", "Amandina", "Arabia", "Comtesse", "Creme brulee",
    "Jelly Black", "Jelly Milk", "Jelly White", "Noblesse", "Noir authentique",
    "Passion au lait", "Stracciatella", "Tentation Noir", "Triangolo"
]

# === Resize utility ===
class ResizeLongestSide:
    def __init__(self, longest=512):
        self.longest = longest

    def __call__(self, img):
        w, h = img.size
        scale = self.longest / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), resample=Image.BILINEAR)

# === Mask transform ===
class MaskTransform:
    def __init__(self):
        self.resize = ResizeLongestSide(512)

    def __call__(self, mask):
        mask = self.resize(mask)
        return torch.as_tensor(np.array(mask), dtype=torch.long)

# === Paths and config ===
data_folder = "data/"
model_path = "model/unet_voc_best.pt"
num_classes = len(VOC_CLASSES)
print("Looking for VOC dataset at:", os.path.abspath(data_folder))

# === Transforms ===
transform = transforms.Compose([
    ResizeLongestSide(512),
    transforms.ToTensor()
])
target_transform = MaskTransform()

# === Load validation dataset ===
val_dataset = datasets.VOCSegmentation(
    data_folder,
    year="2007",
    download=False,
    image_set="val",
    transform=transform,
    target_transform=target_transform,
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
print("Created val_loader")

# === Load model ===
model = UNet(dimensions=num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# === Inference loop ===
y_true = []
y_pred = []

with torch.no_grad():
    for image, mask in val_loader:
        output = model(image)
        pred = output.argmax(dim=1).squeeze().numpy()
        target = mask.squeeze().numpy()

        y_true.extend(target.flatten())
        y_pred.extend(pred.flatten())

# === Compute confusion matrix ===
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

# === Plot raw confusion matrix ===
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=VOC_CLASSES, yticklabels=VOC_CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Raw Confusion Matrix")
plt.tight_layout()
plt.show()

# === Normalize confusion matrix per row ===
cm_normalized = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

# === Plot normalized confusion matrix ===
plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=VOC_CLASSES, yticklabels=VOC_CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix (Per-Class)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
