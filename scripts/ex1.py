import os
os.getcwd()
import numpy as np
import pandas as pd
from si.src.io import csv_file, data_file
from si.src.data import Dataset

# 1.1
iris = read_csv("C:/Users/zemar/si/datasets/iris/iris.csv", features = True, label = True)
iris.summary()


# 1.2 - Dimension of the resulting array after the selection of the penultimate variable
d = iris.X[:,-2].shape[0]
print(d)

# 1.3 - Mean of the last 10 variables
np.nanmean(iris.X[-10:], axis=0)

# 1.4 - Selection of all the samples which values are less than or equal to 6 for all independent variables features
selection = np.all(iris.X <= 6, axis=1)
selected_samples = iris.X[selection]

print(selected_samples.shape[0])

# 1.5 - Selecting all the samples different from 'Iris-setosa'
selected_samples = iris.X[(iris.y != 'Iris-setosa')].shape[0]

print(f'Number of samples: {selected_samples}')
