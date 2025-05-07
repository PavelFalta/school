import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification import bee_dataset
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

ROWS = 4
COLUMNS = 5

# Load via TFDS
ds = tfds.load('bee_dataset/bee_dataset_300',
        batch_size=1,
        as_supervised=True,
        split="train")

import numpy as np
import PIL
from PIL import Image

for example in ds:
    image, label = example
    #print(np.array(label['wasps_output'])[0])
    np.array(image[0])
    Image.fromarray(np.array(image[0]))
    break
    print("=========")

import os
os.makedirs("Dataset/Images", exist_ok=True)

labels = []
for id, example in enumerate(ds):
    Image.fromarray(np.array(image[0])).save(f"Dataset/Images/{id}.jpg")
    labels.append(np.array(label['wasps_output'])[0])
import pandas as pd
pd.DataFrame(labels).to_csv("Dataset/labels.csv")
