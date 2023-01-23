import pandas
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.utils.data as data_utils


sjData = pd.read_csv("../SJData.csv", index_col=[0,1])
sj_train = sjData.head(800).dropna()
sj_validate = sj_train.tail(100)
validate_data = sj_train.drop(['total_cases'], axis = 1)
validate_labels = sj_train.loc[:,['total_cases']]
sj_train = sj_train.head(len(sj_train)-100)
train_data = sj_train.drop(['total_cases'], axis = 1)
train_labels = sj_train.loc[:,['total_cases']]
sj_test = sjData.tail(sjData.shape[0] - 800)
test_data = sj_test.drop(['total_cases'], axis = 1)
test_labels = sj_test.loc[:,['total_cases']]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

input_size = 25 #25 input features
hidden_size = 100
output_size = 1

#num_epochs = 263 #Fine tuned
num_epochs = 600
batch_size = 16
learning_rate = 0.001
#Load the data
train = data_utils.TensorDataset(torch.tensor(train_data.values.astype(np.float32)), torch.tensor(train_labels.values.astype(np.float32)))
test = data_utils.TensorDataset(torch.tensor(test_data.values.astype(np.float32)), torch.tensor(test_labels.values.astype(np.float32)))
validate = data_utils.TensorDataset(torch.tensor(validate_data.values.astype(np.float32)), torch.tensor(validate_labels.values.astype(np.float32)))
train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size)
validate_loader = torch.utils.data.DataLoader(dataset = validate, batch_size = batch_size)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.l_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l_out(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size)

#loss and optimizer
criterion = nn.MSELoss() #Squared euclidean distance, L2 norm
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)
valid_losses = []
for epoch in range(num_epochs):
    for i,(featureValues, labels) in enumerate(train_loader):
        featureValues = featureValues.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(featureValues)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        valid_loss = 0.0
        model.eval()
        for data, labels in validate_loader:
            data, labels = data.to(device), labels.to(device)
            target = model(data)
            loss = criterion(target, labels)
            valid_loss += loss.item()
    valid_losses.append(valid_loss)
    if(epoch%10 == 0):
        print(epoch)
#testing loop
model.eval()
results = [] #(predicted, actual)
with torch.no_grad():
    for (featureValues, labels) in test_loader:
        featureValues = featureValues.to(device)
        labels = labels.to(device)
        outputs = model(featureValues)
        loss = criterion(outputs, labels)
        print('Test Loss: {:.4f}'.format(loss.item()))
        for i in range(len(outputs)):
            results.append((outputs[i], labels[i]))

compare_df = pd.DataFrame(index = sj_test.index, columns=['predicted', 'actual'])
compare_df['predicted'] = [int(x[0]) for x in results]
compare_df['actual'] = [int(x[1]) for x in results]
compare_df.to_csv('compare_df.csv')

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted.plot(ax=axes, label="predicted")
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
#plt.savefig('SJ-FeedForwardNN-263epochs-4Layers')
plt.show()

from sklearn.metrics import r2_score

print(mean_squared_error(compare_df['actual'], compare_df['predicted'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted']))
print(r2_score(compare_df['actual'], compare_df['predicted']))

plt.plot(valid_losses)
plt.savefig("SJ-FeedForwardNN-4 Layers - Validation Loss - 600epochs")
plt.show()

print(min(valid_losses), valid_losses.index(min(valid_losses)))