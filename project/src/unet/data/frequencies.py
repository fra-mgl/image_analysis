import os
import numpy as np
from PIL import Image


import os
import numpy as np
from PIL import Image

def compute_class_frequencies(mask_dir, mask_list, num_classes=14):
    class_counts = np.zeros(num_classes, dtype=np.uint64)
    
    for filename in mask_list:
        mask_path = os.path.join(mask_dir, filename + ".png")
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            for cls in range(num_classes):
                class_counts[cls] += np.sum(mask == cls)
        else:
            print(f"Warning: Mask file not found: {mask_path}")
            
    return class_counts

def load_mask_list(txt_path):
    with open(txt_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]
    return filenames

def compute_class_weights(data_folder, split="train"):
    # Define paths safely
    seg_folder = os.path.join(data_folder, "VOCdevkit", "VOC2007")
    mask_dir = os.path.join(seg_folder, "SegmentationClass")
    txt_path = os.path.join(seg_folder, "ImageSets", "Segmentation", f"{split}.txt")
    
    mask_list = load_mask_list(txt_path)
    counts = compute_class_frequencies(mask_dir, mask_list, num_classes=14)

    weights = 1.0 / (counts + 1e-6)
   
    weights = weights / weights.sum()  # Normalize

    return weights


# if __name__ == "__main__":
#     weights = compute_class_weights()
#     print("Class Weights:", weights)
#     print("Sum of Weights:", weights.sum())