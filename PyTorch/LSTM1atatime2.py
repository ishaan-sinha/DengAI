import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

sjData = pd.read_csv("../SJData.csv", index_col=[1]).loc[:,['total_cases']]
sjData.index = pd.to_datetime(sjData.index)

testDataSize = 139
train_data = sjData[:-testDataSize] #800
test_data = sjData[-testDataSize:] #139

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))

train_data_normalized = scaler.fit_transform(train_data)

train_data_normalized = torch.FloatTensor(train_data_normalized)
train_window = 52

def get_x_y_pairs(train_scaled, train_periods, prediction_periods):

    x_train = [train_scaled[i:i + train_periods] for i in range(len(train_scaled) - train_periods - prediction_periods)]
    y_train = [train_scaled[i + train_periods:i + train_periods + prediction_periods] for i in
               range(len(train_scaled) - train_periods - prediction_periods)]

    x_train = torch.stack(x_train)
    y_train = torch.stack(y_train)

    return x_train, y_train

x_train, y_train = get_x_y_pairs(train_data_normalized, train_window, 1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden == None:
            self.hidden = (torch.zeros(1, 1, self.hidden_size),
                           torch.zeros(1, 1, self.hidden_size))
        else:
            self.hidden = hidden

        lstm_out, self.hidden = self.lstm(x.view(len(x), 1, -1),
                                          self.hidden)

        predictions = self.linear(lstm_out.view(len(x), -1))

        return predictions[-1], self.hidden

model = LSTM(input_size=1, hidden_size=20, output_size=1)
model.to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 2

model.train()

for i in range(epoch):
    for x, y in zip(x_train, y_train):
        x = x.to(device)
        y = y.to(device)
        y_hat, _ = model(x, None)
        optimizer.zero_grad()
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()
    if i % 1 == 0:
        print(i,loss)

test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

for i in range(testDataSize):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        y_hat, _ = model(x, None)
        test_inputs.append(y_hat)

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1,1))

actual_predictions = actual_predictions[:, 0]

pd.DataFrame(actual_predictions).to_csv('LSTM1atatime2-SJ predictions- 2 epochs')

print(mean_squared_error(actual_predictions, test_data['total_cases'], squared=False))
print(mean_absolute_error(actual_predictions, test_data['total_cases']))
print(r2_score(actual_predictions, test_data['total_cases']))