path = "data/data-recovery.csv"

import pandas as pd

df = pd.read_csv(path)

print(df.head())

columns = df.columns

target = "kp0"

for col in columns:
    if target in col:
        print(col)

# 1. 

