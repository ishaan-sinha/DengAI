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
    """
    train_scaled - training sequence
    train_periods - How many data points to use as inputs
    prediction_periods - How many periods to ouput as predictions
    """
    x_train = [train_scaled[i:i + train_periods] for i in range(len(train_scaled) - train_periods - prediction_periods)]
    y_train = [train_scaled[i + train_periods:i + train_periods + prediction_periods] for i in
               range(len(train_scaled) - train_periods - prediction_periods)]
    # -- use the stack function to convert the list of 1D tensors
    # into a 2D tensor where each element of the list is now a row
    x_train = torch.stack(x_train)
    y_train = torch.stack(y_train)

    return x_train, y_train

train_periods = 261
test_periods = 139
prediction_periods = test_periods
x_train, y_train = get_x_y_pairs(train_scaled, train_periods, prediction_periods)

class LSTM(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediction_periods input to get_x_y_pairs
    """

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

model = LSTM(input_size=1, hidden_size=50, output_size=test_periods)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10
model.train()


for epoch in range(epochs + 1):
    for x, y in zip(x_train, y_train):
        x = x.to(device)
        y = y.to(device)
        y_hat, _ = model(x, None)
        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        print(epoch,loss)

model.eval()

with torch.no_grad():
    predictions, _ = model(train_scaled[-train_periods:], None)

#-- Apply inverse transform to undo scaling
predictions = scaler.inverse_transform(np.array(predictions.reshape(-1,1)))
print(predictions)

x = [dt.datetime.date(d) for d in sjData.index]

fig = plt.figure(figsize=(10,5))
plt.title('Dengue Cases')
plt.grid(True)
plt.plot(x[:-len(predictions)],
         sjData.total_cases[:-len(predictions)],
         "b-")
plt.plot(x[-len(predictions):],
         sjData.total_cases[-len(predictions):],
         "b--",
         label='True Values')
plt.plot(x[-len(predictions):],
         predictions,
         "r-",
         label='Predicted Values')
plt.legend()
#plt.savefig('LSTM(261->139), 21epochs')
plt.show()


from sklearn.metrics import r2_score

print(mean_squared_error(predictions, test['total_cases'], squared=False))
print(mean_absolute_error(predictions, test['total_cases']))
print(r2_score(predictions, test['total_cases']))