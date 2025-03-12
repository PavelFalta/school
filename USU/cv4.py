from cv3 import naivni_logisticka_regrese_binarni
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#visualize data
digits = load_digits()


plt.gray()
plt.matshow(digits.images[3])
plt.title(digits.target[3])
plt.show()

#scale data
scaler = StandardScaler()
scaler.fit(digits.data)

X = scaler.transform(digits.data)
Y = digits.target

print(X[0])

# try to clasify one number from rest