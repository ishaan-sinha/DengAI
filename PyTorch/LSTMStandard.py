import torch
import torch.nn as nn
import pandas as pd

sjData = pd.read_csv("../SJData.csv", index_col=[0,1])
train = sjData.head(800).dropna()
train_data = train.drop(['total_cases'], axis = 1)
train_labels = train.loc[:,['total_cases']]
test = sjData.tail(sjData.shape[0] - 800)
test_data = train.drop(['total_cases'], axis = 1)
test_labels = train.loc[:,['total_cases']]
class FluForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FluForecaster, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        output = self.fc(lstm_out[-1])
        return output

# Initialize the model
input_size = 25 #number of features
hidden_size = 64
num_layers = 2
output_size = 1
model = FluForecaster(input_size, hidden_size, num_layers, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    inputs = torch.Tensor(train_data.values)
    targets = torch.Tensor(train_labels.values)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Test the model
with torch.no_grad():
    inputs = torch.Tensor(test_data.values)
    targets = torch.Tensor(test_labels.values)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    print('Test Loss: {:.4f}'.format(loss.item()))
