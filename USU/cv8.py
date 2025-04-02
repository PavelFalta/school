path = "data/data-recovery.csv"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(path)

print(df.head())

columns = df.columns

target = "kp0"

target_col = []

for col in columns:
    if target in col:
        target_col.append(col)

y_data = df[target_col]

print(y_data.head())

Y = y_data[["target_kp0_y", "target_kp0_x"]]

X = y_data.drop(columns=["target_kp0_y", "target_kp0_x"])
X = y_data.drop(columns=["pred_kp0_centroid_y", "pred_kp0_centroid_x", "pred_kp0_sigma_y", "pred_kp0_sigma_x"])

print(X.head())


print(Y.head())



