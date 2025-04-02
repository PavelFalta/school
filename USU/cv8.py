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

# structure 
# 5 predictions
# pred_kp0_pos0_y  pred_kp0_pos0_x
# pred_kp0_pos1_y  pred_kp0_pos1_x
# pred_kp0_pos2_y  pred_kp0_pos2_x
# pred_kp0_pos3_y  pred_kp0_pos3_x
# pred_kp0_pos4_y  pred_kp0_pos4_x
# each has a weight 

# pred_kp0_val0
# pred_kp0_val1
# pred_kp0_val2
# pred_kp0_val3
# pred_kp0_val4

# now want to use random forest regressor to predict the target_kp0_y and target_kp0_x

from sklearn.ensemble import RandomForestRegressor

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(Y.head())



