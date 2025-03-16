import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../../data/data_lab_01/tcga_blood_example.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display
fig, ax = plt.subplots()
ax.imshow(img)
plt.show()