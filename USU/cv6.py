#load from lomy, two folders with images, two classes

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import concurrent.futures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_images_from_folder(folder):
    def load_image(filepath):
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        return img

    images = []
    filepaths = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(load_image, filepaths)
        for img in results:
            if img is not None:
                images.append(img)
    return images

folder1 = 'lomy/stepnylom_jpg'
folder2 = 'lomy/tvarnylom_jpg'

images1 = load_images_from_folder(folder1)
images2 = load_images_from_folder(folder2)
print(len(images1), len(images2))
# for both classes, get colors and display color histograms

def compute_histograms(images):
    histograms = []
    for img in images:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        histograms.append(hist)
    return histograms

histograms1 = compute_histograms(images1)
histograms2 = compute_histograms(images2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(histograms1[0])
plt.subplot(1, 2, 2)
plt.plot(histograms2[0])
plt.show()