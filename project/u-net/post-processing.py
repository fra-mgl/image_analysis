import cv2
import os
import csv
from skimage.io import imread, imshow
from skimage import measure
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk, opening, erosion
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from predict import decode_segmap, VOC_CLASSES, VOC_COLORMAP
from enum import Enum

VOCEnum = Enum('VOCEnum', {name: idx for idx, name in enumerate(VOC_CLASSES)})


def create_legend(classes, colormap, counts):
    legend_patches = [
        Patch(color=np.array(color) / 255.0, label=f"{cls} ({counts[i]})")
        for i, (cls, color) in enumerate(zip(classes, colormap))
    ]
    return legend_patches

VOC_LABELS = [
    "background", 
    "Amandina", 
    "Arabia", 
    "Comtesse", 
    "Creme brulee", 
    "Jelly Black", 
    "Jelly Milk", 
    "Jelly White", 
    "Noblesse", 
    "Noir authentique", 
    "Passion au lait", 
    "Stracciatella", 
    "Tentation Noir", 
    "Triangolo"
]

csv_column_order = [
    "Jelly White", "Jelly Milk", "Jelly Black", "Amandina", "Creme brulee",
    "Triangolo", "Tentation Noir", "Comtesse", "Noblesse", "Noir authentique",
    "Passion au lait", "Arabia", "Stracciatella"
]

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

def count_instances(area, cls):
    # print("Class: ", VOC_LABELS[cls])
    # print("area: ", area)
    # print("res: ", np.round(area / area_threshold[cls], 0))
    # print(cls, area / (area_scaling*area_threshold[cls]))
    return np.round(area / (area_scaling*area_threshold[cls]), 0)

def post_processing(img_path, csv_writer, show):
    # mask = imread(os.path.join("./predict_output", img_path))
    img = Image.open(os.path.join("./predict_output", img_path)).convert('L')
    mask = np.array(img, dtype=np.uint8)
    
    binary_img = mask > 0
    binary_img = opening(binary_img, disk(2))
    binary_img = remove_small_holes(binary_img) 
    binary_img = remove_small_objects(binary_img)

    # binary_img = opening(mask, disk(2))
    # binary_img = closing(mask, disk(1))
    # binary_img = remove_small_holes(binary_img) 
    # binary_img = remove_small_objects(binary_img)

    
    # Label connected components
    label_image, label_num = measure.label(binary_img, return_num=True)

    improved_mask = np.zeros_like(binary_img, dtype=int)

    prediction_counts = np.zeros(14, dtype=int)
    for i in range(1, label_num+1):
        # print("-------")
        region = mask[label_image==i] # .ravel() FIXME is it necessary?
        region_mask = np.where(label_image==i, mask, 0)
        # print(np.unique(region_mask))
        # print(region_mask)
        # class_extract = np.bincount(region).argmax()
        # if class_extract >= 14:
        #     print(class_extract, int(filename.split('.')[0][1:]))
        #     print(np.unique(mask))
        # else:
            # prediction_counts[class_extract] += count_instances(np.count_nonzero(region), class_extract)
            # improved_mask[label_image==i] = class_extract


        class_bins = np.bincount(region)
        for j, b in enumerate(class_bins):
                if j < 14 and j != 0 and b != 0:
                    inst = count_instances(b, j)
                    prediction_counts[j] += inst
                    if inst > 0:
                        improved_mask[region_mask==j] = j            



    # append to csv
    # id_image = int(filename.split('_')[0][1:])
    id_image = int(img_path.split('.')[0][1:])
    row = [id_image]
    for label in csv_column_order:
        index = VOCEnum[label].value
        row.append(int(prediction_counts[index]))
    csv_writer.writerow(row)

    

    if show:
        plt.figure(figsize=(18,8))
        plt.subplot(2, 2, 1)
        plt.imshow(imread(os.path.join("data/VOCdevkit/VOC2007/JPEGImages", 'L'+str(id_image)+'.JPG' )))
        plt.title(f"Input Image {id_image}")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(decode_segmap(mask))
        plt.title("Predicted Segmentation")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(binary_img)
        plt.title(f"Binary Segmentation ({label_num} CC)")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(decode_segmap(improved_mask))
        plt.title("Improved mask")
        plt.axis("off")
         # Add color legend
        legend_patches = create_legend(VOC_CLASSES, VOC_COLORMAP, prediction_counts)
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':

    folder_path = './predict_output'

    with open('submission.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['id', 'Jelly White', 'Jelly Milk', 'Jelly Black', 'Amandina', 'Crème brulée', 'Triangolo', 'Tentation noir', 'Comtesse', 'Noblesse', 'Noir authentique', 'Passion au lait', 'Arabia', 'Stracciatella'])

        # post_processing('L1000944.png', writer, show=True)
        # post_processing('L1010028.png', writer, show=True)
        # post_processing('L1010001.png', writer, show=True)

        # exit(0)

        i = 0
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith('.png'):

                    filename = os.path.basename(entry.path)
                    post_processing(filename, writer, show=False)

                    # i += 1

                    # if i < 20:
                    #     continue
                    # if round(np.random.random(),0):
                    #     filename = os.path.basename(entry.path)
                    #     post_processing(filename, writer, show=True)


                    
                    # if i > 40:
                    #     break
                
                    