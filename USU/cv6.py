#load from lomy, two folders with images, two classes

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def load_images_from_folder(folder):
    images = []
    filepaths = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    for filepath in filepaths:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images

folder1 = 'lomy/stepnylom_jpg'
folder2 = 'lomy/tvarnylom_jpg'

images1 = load_images_from_folder(folder1)
images2 = load_images_from_folder(folder2)
# for both classes, get colors and display color histograms

def get_colors(images):
    colors = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(3):
            colors.append(image[:, :, i].ravel())
    return colors

colors1 = get_colors(images1)
colors2 = get_colors(images2)

def display_histograms(colors):
    for i in range(len(colors)):
        plt.hist(colors[i], bins=256, range=(0, 256), density=True, alpha=0.5)
    plt.show()

display_histograms(colors1)
display_histograms(colors2)