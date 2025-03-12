from cv3 import naivni_logisticka_regrese_binarni
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from collections import Counter

#visualize data
digits = load_digits()


# plt.gray()
# plt.matshow(digits.images[3])
# plt.title(digits.target[3])
# plt.show()

#scale data
scaler = StandardScaler()
scaler.fit(digits.data)

X = scaler.transform(digits.data)
Y = digits.target

# for now, only try to clasify one number from rest

Y = 1 * (Y == 3)

print(Y)
print(Counter(Y))

# train model
model = naivni_logisticka_regrese_binarni()
model.fit(X,Y)

# predict
Y_pred = model.predict(X)

# confusion matrix
cm = confusion_matrix(Y, Y_pred)
print(cm)

plt.matshow(cm, cmap='viridis')
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()

# accuracy
accuracy = accuracy_score(Y, Y_pred)
print(f"Accuracy: {accuracy}")


# try to clasify one number from rest