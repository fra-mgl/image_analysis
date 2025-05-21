"""
Filename: data_format.py
Description: create the dataset
Author: Image-inativi
Date: May 21st 2025
"""
import os
import shutil

#DA LANCIARE DA DENTRO image-analysis/project/u-net . mettere il dataset da convertire qua anche
src_dir = "train_set_not_formatted/train"
voc_root = "data/VOCdevkit/VOC2007"

os.makedirs(f"{voc_root}/JPEGImages", exist_ok=True)
os.makedirs(f"{voc_root}/SegmentationClass", exist_ok=True)
os.makedirs(f"{voc_root}/ImageSets/Segmentation", exist_ok=True)

image_list = []

# code for train or val

for file in os.listdir(src_dir):
    if file.endswith(".jpg"):
        base = file.replace(".jpg", "")
        mask = f"{base}_mask.png"
        img_dst = f"{voc_root}/JPEGImages/{base}.jpg"
        mask_dst = f"{voc_root}/SegmentationClass/{base}.png"
        shutil.copy(os.path.join(src_dir, file), img_dst)
        shutil.copy(os.path.join(src_dir, mask), mask_dst)
        image_list.append(base)

# Create ImageSets/Segmentation/train.txt
with open(f"{voc_root}/ImageSets/Segmentation/train.txt", "w") as f: #train or val
    for img_id in image_list:
        f.write(img_id + "\n")

# code for test
# Copy .jpg files and collect basenames
# for file in os.listdir(src_dir):
#     if file.lower().endswith(".jpg"):
#         base = os.path.splitext(file)[0]  # removes .jpg extension safely
#         img_dst = os.path.join(voc_root, "JPEGImages", f"{base}.jpg")
#         shutil.copy(os.path.join(src_dir, file), img_dst)
#         image_list.append(base)

# # Create test.txt
# test_list_path = os.path.join(voc_root, "ImageSets", "Segmentation", "test.txt") 
# with open(test_list_path, "w") as f:
#     for img_id in image_list:
#         f.write(img_id + "\n")

print(f" Processed {len(image_list)} train images.") #here change val train or test
print(f" Created train.txt at: {voc_root}") #here change val train or test