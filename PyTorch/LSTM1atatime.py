import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

sjData = pd.read_csv("../SJData.csv", index_col=[1]).loc[:,['total_cases']]
sjData.index = pd.to_datetime(sjData.index)

testDataSize = 139
train_data = sjData[:-testDataSize] #800
test_data = sjData[-testDataSize:] #139
#print(len(train_data))
#print(train_data['total_cases'][52])
#print(train_data.columns.get_loc('total_cases'))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))

train_data_normalized = scaler.fit_transform(train_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        #train_label = input_data[i+tw:i+tw+1]
        #train_label = torch.tensor(train_data['total_cases'][i+tw])
        train_label = torch.tensor(train_data_normalized[i+tw][0])
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_periods = 156

train_inout_seq = create_inout_sequences(train_data_normalized, train_periods)


test_periods = 1
prediction_periods = test_periods

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

model = LSTM(input_size=1, output_size=test_periods)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 5
model.train()


for i in range(epochs):
    for seq, labels in train_inout_seq:
        #seq.to(device)
        #labels.to(device)
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(seq)
        single_loss = criterion(y_pred, labels)
        single_loss.backward()
        optimizer.step()


test_inputs = train_data_normalized[-train_periods:].tolist()

model.eval()

for i in range(139):
    seq = torch.FloatTensor(test_inputs[-train_periods:])
    with torch.no_grad():
        predictions, _ = model(test_inputs[-train_periods:], None)
        test_inputs.append(predictions)

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_periods:] ).reshape(-1, 1))

print(actual_predictions)

'''
x = [dt.datetime.date(d) for d in sjData.index]

fig = plt.figure(figsize=(10,5))
plt.title('Dengue Cases')
plt.grid(True)

plt.plot(x[-len(predictions):],
         sjData.total_cases[-len(predictions):],
         "b--",
         label='True Values')
plt.plot(x[-len(predictions):],
         predictions,
         "r-",
         label='Predicted Values')
plt.legend()
plt.savefig('LSTM(156->139), 20epochs, 1 layers')
plt.show()


from sklearn.metrics import r2_score

print(mean_squared_error(predictions, test['total_cases'], squared=False))
print(mean_absolute_error(predictions, test['total_cases']))
print(r2_score(predictions, test['total_cases']))

'''