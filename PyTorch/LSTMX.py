import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


sjData = pd.read_csv("../SJData.csv", index_col=[1]).drop('Unnamed: 0', axis = 1).dropna()
sjData.index = pd.to_datetime(sjData.index)
sjData.to_csv('fullSJData.csv')

testDataSize = 139
train_data = sjData[:-testDataSize] #800
test_data = sjData[-testDataSize:] #139
#print(len(train_data))
#print(train_data['total_cases'][52])
#print(train_data.columns.get_loc('total_cases'))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))

train_data_normalized = scaler.fit_transform(train_data)

#print(train_data_normalized[:5])
#print(train_data_normalized[-5:])

train_data_normalized = torch.FloatTensor(train_data_normalized)

train_window = 52

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        #train_label = input_data[i+tw:i+tw+1]
        #train_label = torch.tensor(train_data['total_cases'][i+tw])
        train_label = torch.tensor(train_data_normalized[i+tw][20])
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

model = LSTM(input_size=26)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 0

for i in range(epochs):
    for seq, labels in train_inout_seq:
        seq.to(device)
        labels.to(device)
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    #if i%20 == 0:
        #print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

#print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = 139
test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())


actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))

actual_predictions = actual_predictions[:, 0]
print(actual_predictions)

'''
pd.DataFrame(actual_predictions).to_csv('LSTMX-SJ predictions- 2 epochs')

print(mean_squared_error(actual_predictions, test_data['total_cases'], squared=False))
print(mean_absolute_error(actual_predictions, test_data['total_cases']))
print(r2_score(actual_predictions, test_data['total_cases']))
'''