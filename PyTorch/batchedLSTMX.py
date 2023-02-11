
import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

sjData = pd.read_csv("../SJData.csv", index_col=[1]).drop('Unnamed: 0', axis = 1).dropna()
move_column = sjData.pop("total_cases")
sjData.insert(0, "total_cases", move_column)

sjData.index = pd.to_datetime(sjData.index)

testDataSize = 139
train_data = sjData[:-testDataSize]
test_data = sjData[-testDataSize:] #139

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))

train_data_normalized = scaler.fit_transform(train_data)
test_data_normalized = scaler.transform(test_data)

