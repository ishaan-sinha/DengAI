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

train = sjData.head(800).dropna() #800
test = sjData.tail(sjData.shape[0] - 800) #139

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
train_scaled = scaler.transform(train)

train_scaled = torch.FloatTensor(train_scaled)
#print(f'Original dimensions : {train_scaled.shape}')
train_scaled = train_scaled.view(-1)
#print(f'Correct dimensions : {train_scaled.shape}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


def get_x_y_pairs(train_scaled, train_periods, prediction_periods):

    x_train = [train_scaled[i:i + train_periods] for i in range(len(train_scaled) - train_periods - prediction_periods)]
    y_train = [train_scaled[i + train_periods:i + train_periods + prediction_periods] for i in
               range(len(train_scaled) - train_periods - prediction_periods)]

    x_train = torch.stack(x_train)
    y_train = torch.stack(y_train)

    return x_train, y_train

train_periods = 156
test_periods = 139
prediction_periods = test_periods
x_train, y_train = get_x_y_pairs(train_scaled, train_periods, prediction_periods)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden = None):
        if hidden == None:
            self.hidden = (torch.zeros(1, 1, self.hidden_size),
                           torch.zeros(1, 1, self.hidden_size))
        else:
            self.hidden = hidden
        # Forward propagate through the first LSTM layer
        lstm_out, self.hidden = self.lstm1(x.view(len(x), 1, -1),
                                          self.hidden)

        # Forward propagate through the second LSTM layer
        lstm_out, self.hidden = self.lstm2(lstm_out.view(len(x), 1, -1),
                                          self.hidden)

        # Decode the hidden state of the last time step
        predictions = self.fc(lstm_out.view(len(x), -1))
        return predictions

model = LSTM(input_size=1, hidden_size=20, output_size=test_periods)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 20
model.train()


for epoch in range(epochs + 1):
    for x, y in zip(x_train, y_train):
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x, None)
        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        print(epoch,loss)

model.eval()

with torch.no_grad():
    predictions = model(train_scaled[-train_periods:], None)

predictions = scaler.inverse_transform(np.array(predictions.reshape(-1,1)))
print(predictions)
print(len(predictions))

from sklearn.metrics import r2_score

print(mean_squared_error(predictions, test['total_cases'], squared=False))
print(mean_absolute_error(predictions, test['total_cases']))
print(r2_score(predictions, test['total_cases']))

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
plt.savefig('LSTM(156->139), 20epochs, 2 layers')
plt.show()


