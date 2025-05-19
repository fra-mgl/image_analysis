import os
import numpy as np
from torchvision import datasets
from torchvision.transforms import Compose
from PIL import Image
from collections import defaultdict
from skimage.measure import regionprops, label
import cv2

def resizing(img):
    w, h = img.size
    scale = 512 / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))    
    return img.resize((new_w, new_h), resample=Image.BILINEAR)


folder_path = './SegmentationClass/'
i = 0
num_classes = 14
class_counts_pixels = np.zeros(num_classes, dtype=np.uint64)
class_counts_instances = {
    1: 41,
    2: 37,
    3: 41,
    4: 32,
    5: 39,
    6: 43,
    7: 47,
    8: 56,
    9: 46,
    10: 46,
    11: 49,
    12: 46,
    13: 46
}

with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith('.png'):
                filename = os.path.basename(entry.path)
                # print(filename)
                img = Image.open(os.path.join(folder_path, filename)).convert('L')

                # resize
                img = resizing(img)
                mask = np.array(img, dtype=np.uint8)

                
                for cls in range(num_classes):
                    class_counts_pixels[cls] += np.sum(mask == cls)
            i += 1
            # if i > 10:
            #     break 

for i in range(1,num_classes):
    print(i, int(class_counts_pixels[i]/(class_counts_instances[i]*3)))

# import pandas as pd
# area_df = pd.DataFrame(dict((f'class_{k}', v) for k, v in area_stats.items()))
# import ace_tools as tools; tools.display_dataframe_to_user(name="Area Statistics by Class", dataframe=area_df)