import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

print(data[:5])
print(target[:5])

print(data.shape)
print(target.shape)

print(raw_df.describe())