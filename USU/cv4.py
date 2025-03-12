from cv3 import naivni_logisticka_regrese_binarni
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


#visualize data
digits = load_digits()
plt.gray()
plt.matshow(digits.images[3])
print(digits.images[3])
#print label
print(digits.target[3])
plt.show()

# try to clasify one number from rest