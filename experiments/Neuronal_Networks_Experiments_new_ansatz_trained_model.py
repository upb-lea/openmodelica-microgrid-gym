import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import zscore
import pandas as pd

np.set_printoptions(precision=20)
import pickle5 as pickle

save_path1 = "C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/net_train_P_f1.pth"
save_path2 = "C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/net_train_P_f2.pth"
save_path3 = "C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/net_train_P_f3.pth"
save_path4 = "C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/net_train_P_f4.pth"
loss_list=[]


#######Restructuring of the data to train the model######
save_path=[save_path1, save_path2, save_path3, save_path4]
class df_transform:
    def __init__(self, df, column_name):
        self.df = df
        self.column_name = column_name

    def create_df(self):
        i = 0
        column_index_boolean = self.df.columns.get_loc(self.column_name)
        index = 0
        for num in column_index_boolean:
            if num == True:
                column_index = index
                break
            index = index + 1

        df_start = pd.Series(self.df.iloc[:, column_index], name=self.column_name)
        df_start = df_start.to_frame()
        df_transformed = df_start
        df_iter = self.df[self.column_name]
        for column in df_iter:
            if i != 0:
                df_aux = pd.Series(df_iter.iloc[:, i])  # assigns the first column = master:freq to df_freq
                df_aux = df_aux.to_frame()
                df_start = pd.concat([df_start, df_aux], join='outer', axis=0, ignore_index=True)
                df_transformed = df_start
            i = i + 1

        return df_transformed

df= pd.read_pickle('Microgrid_Data')
list_columns = ['master.freq', 'master.instPow', 'master.instQ', 'master.voltage', 'slave.freq', 'slave.instPow',
                'slave.instQ', 'slave.voltage', 'load.voltage', 'Load.activePower', 'load.reactivePower']



master_frequency=df_transform(df, 'master.freq')
master_frequency=master_frequency.create_df()
df_training=master_frequency
count=0
for columns in list_columns:
    if count!= 0:
        df_transformed = df_transform(df, columns)
        df_transformed = df_transformed.create_df()
        df_training = pd.concat([df_training, df_transformed], join='outer', axis=1, ignore_index=False)
    count=count+1


#perform z-transformation
df_training_before_z_trafo=df_training.copy()
df_training_before_z_trafo_output=df_training[['master.freq', 'slave.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
df_slave_freq_before_z_trafo = pd.Series(df_training_before_z_trafo['slave.freq'], name='load.freq_false')
df_slave_freq_before_z_trafo = df_slave_freq_before_z_trafo.to_frame()
df_training_before_z_trafo_output=pd.concat([df_training_before_z_trafo_output, df_slave_freq_before_z_trafo], join='outer', axis=1)


df_training_input=df_training[['master.instPow', 'slave.instPow', 'master.instQ', 'slave.instQ', 'Load.activePower', 'load.reactivePower']]
df_slave_freq=df_training['slave.freq']
df_slave_freq = pd.Series(df_slave_freq, name='load.freq_false')
df_slave_freq = df_slave_freq.to_frame()
df_training_output=df_training[['master.freq', 'slave.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
df_training_output=pd.concat([df_training_output, df_slave_freq], join='outer', axis=1)
for columns2 in list_columns:
    df_training[columns2] = zscore(df_training[columns2])


df_training_input=df_training[['master.instPow', 'slave.instPow', 'master.instQ', 'slave.instQ', 'Load.activePower', 'load.reactivePower']]
df_slave_freq=df_training['slave.freq']
df_slave_freq = pd.Series(df_slave_freq, name='load.freq_false')
df_slave_freq = df_slave_freq.to_frame()
#df_slave_freq.rename(columns={'slave.freq': 'load.freq_false'}, inplace=True)
df_training_output=df_training[['master.freq', 'slave.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
df_training_output=pd.concat([df_training_output, df_slave_freq], join='outer', axis=1)
df_training=pd.concat([df_training_input, df_training_output], join='outer', axis=1)
x=1

# dataset definition
class CSVDataset:
    # load the dataset
    def __init__(self, df):
        # store the inputs and outputs
        self.X = df.iloc[:, 0:6]  # one dimensional input: power
        self.y = df.iloc[:, 6:12]  # one dimensional input: frequency
        self.X = torch.from_numpy(self.X.to_numpy()).float()
        self.y = torch.from_numpy(self.y.to_numpy()).float()

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


#Load Train Dataset#
train = CSVDataset(df_training)
train_dl = torch.utils.data.DataLoader(train, batch_size=20, shuffle=True)


number_epochs = 15
list_neurons= [8, 16, 32, 64]
a=0
for neurons in list_neurons:

####Definition of the feed-forward-neural network###
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(6, neurons)
            self.fc2 = nn.Linear(neurons, neurons*2)
            self.fc3 = nn.Linear(neurons*2, neurons*2)
            self.fc4 = nn.Linear(neurons, 6)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)


    net = Net()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_epoch_loss = 0

    loss_stats = {
        'train': []
     }

    for epoch in range(number_epochs):
        train_epoch_loss = 0
        for i, (inputs, targets) in enumerate(tqdm(train_dl)):
            # inputs = inputs.view(-1, 6)
            y_pred = net(inputs)  # x_train:batch_sizexfeature_dimension
            b = targets.size()
            y_pred = torch.squeeze(y_pred)
            train_loss = criterion(y_pred, targets)  # Vektor der größe batch_size # shuffle ich nach jeder epoche
            # print(train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
        loss_stats['train'].append(train_epoch_loss / len(train_dl))

    plt.plot(range(number_epochs), loss_stats['train'])
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.savefig(
        "C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/plots/SSNN{}.png".format(
            neurons))
    print('Training process has finished.')
    torch.save(net.state_dict(), save_path[a])
    loss_list.append(loss_stats)
    a=a+1



def retransformation_zscore(array, df):
    list_values=[]
    i=0# Function that calculates the retransformation of the z score
    for column in df:
        x = array[i] * df[column].std() + df[column].mean()  # Z=X-mü/sigma || X=Z*sigma + mü
        list_values.append(x)
        i=i+1
    return list_values

# Print about testing
print('Starting testing')

# Saving the model
save_path = './net_train_P_f.pth'
torch.save(net.state_dict(), save_path)


