import cv2
import os
import csv
from skimage.io import imread, imshow
from skimage import measure
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk, opening, erosion
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

def post_processing(img_path, csv_writer, show):
    mask = imread(os.path.join("./predict_output", img_path))
    
    binary_img = mask > 0
    binary_img = remove_small_holes(binary_img) 
    binary_img = remove_small_objects(binary_img)

    # Label connected components
    label_image, label_num = measure.label(binary_img, return_num=True)

    improved_mask = np.zeros_like(binary_img, dtype=int)

    prediction_counts = np.zeros(14, dtype=int)
    for i in range(1, label_num+1):
        region = mask[label_image==i].ravel()
        class_extract = np.bincount(region).argmax()
        if class_extract >= 14:
            print(class_extract, int(filename.split('.')[0][1:]))
            print(np.unique(mask))
        else:
            prediction_counts[class_extract] += 1        
            improved_mask[label_image==i] = class_extract


    # append to csv
    # id_image = int(filename.split('_')[0][1:])
    id_image = int(filename.split('.')[0][1:])
    row = [id_image]
    for label in csv_column_order:
        index = VOCEnum[label].value
        row.append(int(prediction_counts[index]))
    csv_writer.writerow(row)

    

    if show:
        plt.figure(figsize=(20,8))
        plt.subplot(2, 2, 1)
        plt.imshow(imread(os.path.join("data/VOCdevkit/VOC2007/JPEGImagesTest/test", img_path )))
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

        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith('.jpg'):
                    filename = os.path.basename(entry.path)
                    post_processing(filename, writer, show=True)
                
                    