# import torch
# import torchvision.transforms.v2 as T
# from torchvision.transforms.v2 import functional as F
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# class CutoutTransform(torch.nn.Module):
#     def __init__(self, num_holes=4, size=100, fill_value=0):
#         super().__init__()
#         self.num_holes = num_holes
#         self.size = size
#         self.fill_value = fill_value

#     def forward(self, image, mask):
#         image_np = np.array(image)
#         mask_np = np.array(mask)

#         h, w = image_np.shape[:2]

#         # Find all non-zero pixels in the mask
#         nonzero_coords = np.argwhere(mask_np > 0)
#         if len(nonzero_coords) == 0:
#             # Fallback: no valid non-zero pixels
#             return image, mask

#         for _ in range(self.num_holes):
#             # Randomly choose a non-zero pixel to center the cutout
#             center_y, center_x = nonzero_coords[np.random.choice(len(nonzero_coords))]

#             # Compute top-left corner of cutout region
#             top = max(center_y - self.size // 2, 0)
#             left = max(center_x - self.size // 2, 0)

#             # Ensure cutout stays within image bounds
#             bottom = min(top + self.size, h)
#             right = min(left + self.size, w)
#             top = bottom - self.size
#             left = right - self.size

#             # Apply cutout to image and mask
#             image_np[top:bottom, left:right] = self.fill_value
#             mask_np[top:bottom, left:right] = 0

#         return Image.fromarray(image_np), Image.fromarray(mask_np)


# # Load images
# image = Image.open("image.jpg").convert("RGB")
# mask = Image.open("mask.png").convert("L")

# # Apply transform
# cutout = CutoutTransform()
# image_c, mask_c = cutout(image, mask)

# # Show
# plt.subplot(1,2,1)
# plt.imshow(image_c)
# plt.title("Image")

# plt.subplot(1,2,2)
# plt.imshow(mask_c, cmap="gray")
# plt.title("Mask")
# plt.show()

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch

# Define the custom cutout transform
class CutoutTransform(torch.nn.Module):
    def __init__(self, num_holes=4, size=200, fill_value=0):
        super().__init__()
        self.num_holes = num_holes
        self.size = size
        self.fill_value = fill_value

    def forward(self, image, mask):
        image_np = np.array(image)
        mask_np = np.array(mask)

        h, w = image_np.shape[:2]
        nonzero_coords = np.argwhere(mask_np > 0)
        if len(nonzero_coords) == 0:
            return image, mask

        for _ in range(self.num_holes):
            center_y, center_x = nonzero_coords[np.random.choice(len(nonzero_coords))]
            top = max(center_y - self.size // 2, 0)
            left = max(center_x - self.size // 2, 0)
            bottom = min(top + self.size, h)
            right = min(left + self.size, w)
            top = bottom - self.size
            left = right - self.size

            image_np[top:bottom, left:right] = self.fill_value
            mask_np[top:bottom, left:right] = 0

        return Image.fromarray(image_np), Image.fromarray(mask_np)

# Paths
base_path = "VOCdevkit/VOC2007"
image_dir = os.path.join(base_path, "JPEGImages")
mask_dir = os.path.join(base_path, "SegmentationClass")
out_image_dir = os.path.join(base_path, "CutoutImages")
out_mask_dir = os.path.join(base_path, "CutoutMasks")
cutoff_list_path = os.path.join(base_path, "train_cutoff.txt")

os.makedirs(out_image_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

# Transformation
cutout = CutoutTransform()

# Open file to write processed filenames
with open(cutoff_list_path, "w") as list_file:
    for filename in tqdm(os.listdir(image_dir), desc="Processing images"):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace(".jpg", ".png"))

        if not os.path.exists(mask_path):
            tqdm.write(f"[SKIP] Mask not found for {filename}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            tqdm.write(f"[LOAD] {filename} and corresponding mask loaded")

            image_cutout, mask_cutout = cutout(image, mask)

            image_cutout.save(os.path.join(out_image_dir, filename))
            mask_cutout.save(os.path.join(out_mask_dir, filename.replace(".jpg", ".png")))

            # Write the filename without extension to the cutoff list
            basename = os.path.splitext(filename)[0]
            list_file.write(basename + "\n")

            tqdm.write(f"[SAVE] Processed {filename}")
        except Exception as e:
            tqdm.write(f"[ERROR] Failed on {filename}: {e}")
