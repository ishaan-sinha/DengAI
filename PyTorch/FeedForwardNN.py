import pandas
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

sjData = pd.read_csv("../SJData.csv", index_col=[0,1])
#device config

sj_subtrain = sjData.head(800).dropna()

target = torch.tensor(sj_subtrain['total_cases'].values.astype(np.float32))
features = torch.tensor(sj_subtrain.drop(['total_cases'], axis = 1).values.astype(np.float32))
target = target.type(torch.FloatTensor)

train = torch.utils.data.TensorDataset(features, target)

sj_subtest = sjData.tail(sjData.shape[0] - 800)
testTarget = torch.tensor(sj_subtest['total_cases'].values.astype(np.float32))
featuresTest = torch.tensor(sj_subtest.drop(['total_cases'], axis = 1).values.astype(np.float32))
testTarget = testTarget.type(torch.FloatTensor)

test = torch.utils.data.TensorDataset(featuresTest, testTarget)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

input_size = 25 #25 input features
hidden_size = 100
output_size = 1

num_epochs = 10
batch_size = 16
learning_rate = 0.001

#Load the data

train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size)


examples = iter(train_loader)
samples, labels = examples.next()

#print(samples.shape, labels.shape)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.l_out = nn.Linear(hidden_size, output_size)
        #self.l_out
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l_out(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size)

#loss and optimizer
criterion = nn.MSELoss() #Squared euclidean distance, L2 norm
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (featureValues, labels) in enumerate(train_loader):
        featureValues = featureValues.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(featureValues)
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%16 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#testing loop
model.eval()
results = [] #(predicted, actual)
with torch.no_grad():

    for featureValues, labels in test_loader:
        labels = labels.to(device)
        outputs = model(featureValues)

        #value, index
        _, predictions = torch.max(outputs, 1)
        #n_samples += labels.shape[0]
        #n_correct += (predictions==labels).sum().item()
        for i in range(len(predictions)):
            results.append((predictions[i], labels[i]))
    #Mape, Rmse, r^2
    #print(f'Accuracy of the network on the test times: {acc} %')

compare_df = pd.DataFrame(index = sj_subtest.index, columns=['predicted', 'actual'])
compare_df['predicted'] = [int(x[0]) for x in results]
compare_df['actual'] = [int(x[1]) for x in results]
compare_df.to_csv('compare_df.csv')

plt.clf()
figs, axes = plt.subplots(nrows=1, ncols=1)
compare_df.actual.plot(ax=axes, label="actual")
compare_df.predicted.plot(ax=axes, label="predicted")
plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()
#plt.savefig('SJ-FeedForwardNN')
plt.show()

from sklearn.metrics import r2_score

print(mean_squared_error(compare_df['actual'], compare_df['predicted'], squared=False))
print(mean_absolute_error(compare_df['actual'], compare_df['predicted']))
print(r2_score(compare_df['actual'], compare_df['predicted']))