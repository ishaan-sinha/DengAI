
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

sj_train = sjData[:-testDataSize]

target = 'total_cases'
target_mean = sj_train[target].mean()
target_stdev = sj_train[target].std()

sj_test = sjData[-testDataSize:] #139

for c in sj_train.columns:
    mean = sj_train[c].mean()
    stdev = sj_train[c].std()

    sj_train[c] = (sj_train[c] - mean) / stdev
    sj_test[c] = (sj_test[c] - mean) / stdev

import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, sequence_length=5):
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe.loc[:, dataframe.columns != target].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

sequence_length = 8
batch_size = 16

train_dataset = SequenceDataset(
    sj_train,
    target='total_cases',
    sequence_length=sequence_length
)

test_dataset = SequenceDataset(
    sj_test,
    target='total_cases',
    sequence_length=sequence_length
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X, y = next(iter(train_loader))

#print("Features shape:", X.shape) #[16, 8, 25]
#print("Target shape:", y.shape) #[16]

from torch import nn


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

learning_rate = .001
num_hidden_units = 100

model = ShallowRegressionLSTM(num_sensors=len(sjData.axes[1]) - 1, hidden_units=num_hidden_units)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_loss = []
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    train_loss.append(avg_loss)
    print(f"Train loss: {avg_loss}")


test_loss = []
def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    test_loss.append(avg_loss)
    print(f"Test loss: {avg_loss}")


print("Untrained test\n--------")
test_model(test_loader, model, loss_function)
print()

epochs = 75

for ix_epoch in range(epochs):
    print(f"Epoch {ix_epoch}\n---------")
    train_model(train_loader, model, loss_function, optimizer=optimizer)
    test_model(test_loader, model, loss_function)
    print()

results = [] #(predicted, actual)
model.eval()
with torch.no_grad():
    for X, y in test_loader:
        output = model(X)
        for i in range(len(output)):
            results.append((output[i]*target_stdev+target_mean, y[i]*target_stdev + target_mean))
compare_df = pd.DataFrame(index = sj_test.index, columns=['predicted', 'actual'])
compare_df['predicted'] = [int(x[0]) for x in results]
compare_df['actual'] = [int(x[1]) for x in results]

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted.plot(ax=axes, label="predicted")
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
plt.savefig('TOTALBatchedLSTMX-75epochs + Weightdecay + 1atatime')
plt.show()


print(mean_squared_error(compare_df['actual'], compare_df['predicted'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted']))
print(r2_score(compare_df['actual'], compare_df['predicted']))


'''

print(len(train_loss))
plt.plot(train_loss)
plt.savefig("BatchedLSTM2layer1atatime - Train Loss - 200epochs")
plt.show()

print(len(test_loss))
plt.plot(test_loss)
plt.savefig("BatchedLSTM2layeratatime - Test Loss - 200epochs")
plt.show()
'''