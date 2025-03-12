from cv3 import naivni_logisticka_regrese_binarni
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

#visualize data
digits = load_digits()


# plt.gray()
# plt.matshow(digits.images[3])
# plt.title(digits.target[3])
# plt.show()

scaler = MinMaxScaler()
scaler.fit(digits.data)

X = scaler.transform(digits.data)
Y = digits.target

# for now, only try to classify one number from rest
Y = 1 * (Y == 3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = naivni_logisticka_regrese_binarni()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

cm = confusion_matrix(Y_test, Y_pred)
print(cm)

report = classification_report(Y_test, Y_pred)
print(report)

# plt.matshow(cm, cmap='viridis')
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.show()


# digits = load_digits()
# X = digits.data
# Y = digits.target

# # Initialize models
# models = [LogisticRegression(max_iter=10000) for _ in range(10)]

# # Fit models
# for i in range(10):
#     binary_target = (Y == i).astype(int)
#     X_train, X_test, Y_train, Y_test = train_test_split(X, binary_target, test_size=0.3, random_state=42)
#     models[i].fit(X_train, Y_train)

# # Predict using models
# Y_pred_probas = np.array([model.predict_proba(X_test)[:, 1] for model in models]).T
# Y_pred = np.argmax(Y_pred_probas, axis=1)

# # Compute and display confusion matrix and classification report
# cm = confusion_matrix(Y_test, Y_pred)
# print("Confusion Matrix:\n", cm)

# report = classification_report(Y_test, Y_pred)
# print("Classification Report:\n", report)