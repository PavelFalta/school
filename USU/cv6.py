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

def get_grayscale_values(images):
    grayscale_values = []
    for idx, image in enumerate(images):
        # print(f"Processing image {idx + 1}/{len(images)} for grayscale extraction.")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_values.extend(gray_image.ravel())
    print("Grayscale extraction completed.")
    return grayscale_values

grayscale_values1 = get_grayscale_values(images1)
grayscale_values2 = get_grayscale_values(images2)

def display_grayscale_histogram(values, label):
    print(f"Displaying grayscale histogram for {label}.")
    plt.hist(values, bins=256, range=(0, 256), density=True, color='gray', alpha=0.7, label=label)
    plt.legend()
    plt.show()
    print(f"Grayscale histogram for {label} displayed.")

display_grayscale_histogram(grayscale_values1, "Class 1")
display_grayscale_histogram(grayscale_values2, "Class 2")
