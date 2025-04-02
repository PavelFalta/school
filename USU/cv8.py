path = "data/data-recovery.csv"

import pandas as pd

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

# 1. 

