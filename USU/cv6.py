#load from lomy, two folders with images, two classes

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import concurrent.futures

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

def get_colors_optimized(images):
    colors = [[], [], []]
    for idx, image in enumerate(images):
        print(f"Processing image {idx + 1}/{len(images)} for color extraction.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(3):
            colors[i].extend(image[:, :, i].ravel())
    print("Color extraction completed.")
    return colors

colors1 = get_colors_optimized(images1)
colors2 = get_colors_optimized(images2)

def display_histograms_optimized(colors):
    colors_labels = ['Red', 'Green', 'Blue']
    print("Displaying histograms.")
    for i in range(3):
        print(f"Plotting {colors_labels[i]} histogram.")
        plt.hist(colors[i], bins=256, range=(0, 256), density=True, alpha=0.5, label=colors_labels[i])
    plt.legend()
    plt.show()
    print("Histograms displayed.")

display_histograms_optimized(colors1)
display_histograms_optimized(colors2)