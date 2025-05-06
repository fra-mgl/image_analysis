import os
import shutil

src_dir = "."
voc_root = "VOCdevkit/VOC2007"

os.makedirs(f"{voc_root}/JPEGImages", exist_ok=True)
os.makedirs(f"{voc_root}/SegmentationClass", exist_ok=True)

image_list = []

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
with open(f"{voc_root}/ImageSets/Segmentation/train.txt", "w") as f:
    for img_id in image_list:
        f.write(img_id + "\n")
