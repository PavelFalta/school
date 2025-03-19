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

def get_colors_concurrent(images):
    def extract_colors(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return [image[:, :, i].ravel() for i in range(3)]

    colors = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(extract_colors, images)
        for color_set in results:
            colors.extend(color_set)
    return colors

colors1 = get_colors_concurrent(images1)
colors2 = get_colors_concurrent(images2)

def display_histograms_concurrent(colors):
    def plot_histogram(color):
        plt.hist(color, bins=256, range=(0, 256), density=True, alpha=0.5)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(plot_histogram, colors)
    plt.show()

display_histograms_concurrent(colors1)
display_histograms_concurrent(colors2)