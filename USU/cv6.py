

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import concurrent.futures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')


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


def compute_histograms(images):
    histograms = []
    for img in images:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        histograms.append(hist)
    return histograms

histograms1 = compute_histograms(images1)
histograms2 = compute_histograms(images2)










X = np.array([hist.flatten() for hist in histograms1 + histograms2])
y = np.array([0] * len(histograms1) + [1] * len(histograms2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





kf = KFold(n_splits=5, shuffle=True, random_state=42)

def train_xgboost_kfold(n_estimators, max_depth, learning_rate, min_child_weight):
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        clf = XGBClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            min_child_weight=int(min_child_weight),
            use_label_encoder=False,
            eval_metric='logloss'
        )
        clf.fit(X_train_fold, y_train_fold)
        y_pred_fold = clf.predict(X_test_fold)
        accuracies.append(np.mean(y_pred_fold == y_test_fold))
    
    return np.mean(accuracies)


pbounds = {
    'n_estimators': (10, 1000),
    'max_depth': (1, 100),
    'learning_rate': (0.01, 0.3),
    'min_child_weight': (1, 10)
}

optimizer = BayesianOptimization(f=train_xgboost_kfold, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=20, n_iter=20)

best = optimizer.max['params']


clf = XGBClassifier(
    n_estimators=int(best['n_estimators']),
    max_depth=int(best['max_depth']),
    learning_rate=best['learning_rate'],
    min_child_weight=int(best['min_child_weight']),
    use_label_encoder=False,
    eval_metric='logloss'
)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()
