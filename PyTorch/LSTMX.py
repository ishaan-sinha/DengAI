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
#print(sjData.columns) #679x26
testDataSize = 139
train_data = sjData[:-testDataSize]
test_data = sjData[-testDataSize:] #139

#print(sjData.columns.get_loc("total_cases")) 20th column

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))

train_data_normalized = scaler.fit_transform(train_data)
test_data_normalized = scaler.fit_transform(test_data.drop('total_cases', axis = 1))

saved = train_data_normalized.copy()

train_data_normalized = torch.FloatTensor(train_data_normalized)

train_window = 52

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw-1):
        #train_seq = input_data[i:i+tw]
        train_seq = [input_data[j] for j in range(i, i+tw)]
        x_train = torch.stack(train_seq)
        train_label = (torch.tensor(saved[i+tw][0])).type('torch.FloatTensor')
        inout_seq.append((x_train, train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

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


model = LSTM(input_size=26, hidden_size=20, output_size=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 0

model.train()

for i in range(epochs):
    for seq, labels in train_inout_seq:
        seq.to(device)
        labels.to(device)
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))

        y_pred, _ = model(seq, None)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%1 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')


fut_pred = 139


model.eval()

test_inputs = train_data_normalized[-train_window:]

print(type(test_data_normalized[1]))
print(test_data_normalized[1])

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        train_seq = [test_inputs[j] for j in range(i, i + train_window)]
        x_train = torch.stack(train_seq)
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        y_hat, _ = model(x_train, None)
        original = test_data_normalized[i]
        toAdd = np.insert(original, 0, y_hat)
        toAdd = torch.FloatTensor(toAdd)
        test_inputs = torch.cat([test_inputs, toAdd])

'''
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))

actual_predictions = actual_predictions[:, 0]
print(actual_predictions)


pd.DataFrame(actual_predictions).to_csv('LSTMX-SJ predictions- 2 epochs')

print(mean_squared_error(actual_predictions, test_data['total_cases'], squared=False))
print(mean_absolute_error(actual_predictions, test_data['total_cases']))
print(r2_score(actual_predictions, test_data['total_cases']))
'''