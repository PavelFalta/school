#load from lomy, two folders with images, two classes

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import concurrent.futures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from bayes_opt import BayesianOptimization


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

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(histograms1[0])
# plt.subplot(1, 2, 2)
# plt.plot(histograms2[0])
# plt.show()

# create dataset

X = np.array([hist.flatten() for hist in histograms1 + histograms2])
y = np.array([0] * len(histograms1) + [1] * len(histograms2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train random forest classifier

# bayes optimization

def train_random_forest(n_estimators, max_depth):
    clf = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return np.mean(y_pred == y_test)

pbounds = {
    'n_estimators': (10, 1000),
    'max_depth': (1, 100),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20)
}

def train_random_forest(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf)
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return np.mean(y_pred == y_test)

optimizer = BayesianOptimization(f=train_random_forest, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=20, n_iter=20)

best = optimizer.max['params']

clf = RandomForestClassifier(
    n_estimators=int(best['n_estimators']),
    max_depth=int(best['max_depth']),
    min_samples_split=int(best['min_samples_split']),
    min_samples_leaf=int(best['min_samples_leaf'])
)
clf.fit(X_train, y_train)

# evaluate classifier

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()