#load from lomy, two folders with images, two classes

import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = np.loadtxt(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

folder1 = 'lomy/stepnylom_jpg'
folder2 = 'lomy/tvarnylom_jpg'

images1 = load_images_from_folder(folder1)
images2 = load_images_from_folder(folder2)
print(len(images1))
print(len(images2))