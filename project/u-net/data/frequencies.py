import os
import numpy as np
from PIL import Image


def compute_class_frequencies(mask_dir, num_classes=14):
    class_counts = np.zeros(num_classes, dtype=np.uint64)
    for filename in os.listdir(mask_dir):
        if filename.endswith('.png'):
            mask = np.array(Image.open(os.path.join(mask_dir, filename)))
            for cls in range(num_classes):
                class_counts[cls] += np.sum(mask == cls)
    return class_counts


def compute_class_weights():

    counts = compute_class_frequencies("data/VOCdevkit/VOC2007/SegmentationClass", num_classes=14)
    weights = 1.0 / (counts + 1e-6)
    weights[0] = weights[1:].min()  # match it to the rarest non-background class
    weights[1] *= 2.0
    weights = weights / weights.sum()  # Normalize
   

    return weights

# if __name__ == "__main__":
#     weights = compute_class_weights()
#     print("Class Weights:", weights)
#     print("Sum of Weights:", weights.sum())