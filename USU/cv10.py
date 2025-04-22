import keras
from keras import models
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

path = 'data/kopisty_pocasi_rozsireno.csv'
df = pd.read_csv(path, delimiter = ';', decimal = ',')
# print(df.head())

# dest is 1 if uhrn_srazky_1 or uhrn_srazky_2 is more than 0
df["dest"] = (df["uhrn_srazky_1"] > 0) | (df["uhrn_srazky_2"] > 0)

# now need to shift dest by 1 day
df["dest_zitra"] = df["dest"].shift(-1)
# this is how it looks 01.09.2019
df["mesic"] = df["datum"].str[3:5]


# print(df.head())
# now need to encode dest_zitra as 0 or 1
le = LabelEncoder()
df["dest_zitra"] = le.fit_transform(df["dest_zitra"])
print(df.head())

#         datum  teplota  uhrn_srazky_1  rychlost_vitr  max_naraz_vitr  vlhkost  vypar  uhrn_srazky_2         tlak   dest  dest_zitra mesic
# 0  01.09.2019     19.2           20.1            0.6            10.8       94      0           12.6  1007.116499   True           0    09
# 1  02.09.2019     15.1            0.0            2.0            11.4       64      0            0.0  1016.544746  False           0    09
# 2  03.09.2019     15.3            0.0            0.8             6.4       70      0            0.0  1019.923534  False           0    09
# 3  04.09.2019     17.0            0.0            0.9             8.0       70      0            0.0  1013.585624  False           0    09
# 4  05.09.2019     15.8            0.0            1.9            11.4       67      0            0.0  1013.116874  False           1    09

X = df[["mesic", "teplota", "uhrn_srazky_1", "rychlost_vitr", "max_naraz_vitr", "vlhkost", "vypar", "uhrn_srazky_2", "tlak"]]
y = df["dest_zitra"]

print(X.head())
print(y.head())

# now need to split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train.head())
# print(y_train.head())

# now need to scale the data
scaler = StandardScaler()
# scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(X_train.head())
# print(X_test.head())

# now need to build the model
model = models.Sequential()
model.add(layers.Dense(units = 32, input_shape = X_train.shape[1:], activation = 'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units = 16, activation = 'relu'))
model.add(layers.Dense(units = 1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 1000, batch_size = 32)

# now need to evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# plot the history
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Accuracy and Loss')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='upper left')
plt.show()

# plot the history
# plt.plot(history.history['loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.show()
