import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.jit as jit
from torch.nn import Parameter as TorchParam
from torch import Tensor
from typing import List, Tuple
import torch.jit as jit
from scipy.stats import zscore
import pandas as pd

# vergangenheitswerte noch mit reingeben
np.set_printoptions(precision=20)
import pickle5 as pickle

save_path1 = "C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/net_train_P_f1_exp_moreData.pth"
save_path2 = "C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/net_train_P_f2_exp.pth"
save_path3 = "C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/net_train_P_f3_exp.pth"
save_path4 = "C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/net_train_P_f4_exp.pth"
loss_list = []
#####Organize the data to train the model#####
save_path = [save_path1, save_path2, save_path3, save_path4]
ts = 5e-4


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


# df_training_prep= pd.read_pickle('Microgrid_Data_new_training_validation')
df_all_data = pd.read_pickle('Microgrid_Data_stochastic_component')
#df_all_data=pd.read_pickle('Microgrid_data_less_fluc')

list_columns = ['master.freq', 'master.instPow', 'master.instQ', 'master.voltage', 'slave.freq', 'slave.instPow',
                'slave.instQ', 'slave.voltage', 'load.freq', 'Load.activePower', 'load.reactivePower', 'load.voltage']
df_validation_prep = pd.read_pickle('Microgrid_Data_validation')
# df_validation_prep=df_validation_prep.iloc[:,2880:]

#df_test_prep = pd.read_pickle('Microgrid_Data_stochastic_test')


##############################################Trainings_set#############################################################
def df_second_restruct(df_prep):
    master_frequency = df_transform(df_prep, 'master.freq')
    master_frequency = master_frequency.create_df()
    df_sec_trafo = master_frequency
    count = 0
    for columns in list_columns:
        if count != 0:
            df_transformed = df_transform(df_prep, columns)
            df_transformed = df_transformed.create_df()
            df_sec_trafo = pd.concat([df_sec_trafo, df_transformed], join='outer', axis=1, ignore_index=False)
        count = count + 1
    df_sec_trafo = df_sec_trafo.dropna()
    df_sec_trafo.reset_index(inplace=True)
    df_sec_trafo.drop('index', axis='columns', inplace=True)
    return df_sec_trafo


df_all_data = df_second_restruct(df_all_data)
df_all_data_before_trafo_output=df_all_data[
    ['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]

df_training=df_all_data.iloc[:750000,:]
df_validation=df_all_data.iloc[750000:885000,:]
df_test = df_all_data.iloc[885000:,:]

# perform z-transformation --> can be shorted later, when the real frequency on the load bus is used
df_training_before_trafo_output = df_training[
    ['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
# df_validation_before_trafo_output=df_validation[['master.freq', 'slave.freq', 'master.voltage', 'slave.voltage', 'load.voltage', 'load.freq']]
df_test_before_trafo_output = df_test[
    ['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
df_test_before_trafo_input = df_test[
    ['master.instPow', 'slave.instPow', 'Load.activePower', 'master.instQ', 'slave.instQ', 'load.reactivePower']]
df_validation_before_trafo_output = df_validation[
    ['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
df_validation_before_trafo_input = df_validation[
    ['master.instPow', 'slave.instPow', 'Load.activePower', 'master.instQ', 'slave.instQ', 'load.reactivePower']]


# df_validation_before_z_trafo=df_validation.copy()
# # df_validation_before_z_trafo_output=df_validation[['master.freq', 'slave.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
# # df_slave_freq_before_z_trafo = pd.Series(df_validation_before_z_trafo['slave.freq'], name='load.freq_false')
# # df_slave_freq_before_z_trafo = df_slave_freq_before_z_trafo.to_frame()
# # df_validation_before_z_trafo_output=pd.concat([df_validation_before_z_trafo_output, df_slave_freq_before_z_trafo], join='outer', axis=1)


def df_min_max_transformation(df):
    for columns2 in list_columns:
        df[columns2] = (df[columns2] - df[columns2].min()) / (df[columns2].max() - df[columns2].min())
    return df


df_all_data = df_min_max_transformation(df_all_data)
df_training=df_all_data.iloc[:750000,:]
df_validation=df_all_data.iloc[750000:885000,:]
df_test = df_all_data.iloc[885000:,:]
#df_test = df_min_max_transformation(df_test)
df_test_output = df_test[['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]

# def retransformation_min_max_scaling(df_before_trafo, df):
#     df_new = df_before_trafo * (df['slave.freq'].max() - df['slave.freq'].min()) + df['slave.freq'].min()  # Z=X-mü/sigma || X=Z*sigma + mü
#     return df_new

# testfreq=retransformation_min_max_scaling(df_training_before_trafo_output['slave.freq'], df_training)
# x=1


####################Definition of the ANN###############################################################################
########################################################################################################################


# dataset definition
class CSVDataset:
    # load the dataset
    def __init__(self, df_in, df_out):
        # store the inputs and outputs
        self.X = df_in.to_numpy()  # one dimensional input: power
        self.y = df_out.to_numpy()

        # self.X = torch.from_numpy(self.X.to_numpy()).float()
        # self.y = torch.from_numpy(self.y.to_numpy()).float()

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


df_training_input = df_training[
    ['master.instPow', 'slave.instPow', 'Load.activePower', 'master.instQ', 'slave.instQ', 'load.reactivePower']]
df_training_output = df_training[
    ['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
train = CSVDataset(df_training_input, df_training_output)

df_validation_input=df_validation[['master.instPow', 'slave.instPow', 'Load.activePower', 'master.instQ', 'slave.instQ','load.reactivePower']]
df_validation_output = df_validation[['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
validation = CSVDataset(df_validation_input, df_validation_output)

df_test_input = df_test[
    ['master.instPow', 'slave.instPow', 'Load.activePower', 'master.instQ', 'slave.instQ', 'load.reactivePower']]
df_test_output = df_test[['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
test = CSVDataset(df_test_input, df_test_output)


# test_dl = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)


def generate_tensor_train(dataset_train):
    tensorX_train = np.full((3000, 250, 6), np.nan)
    tensorY_train = np.full((3000, 250, 6), np.nan)
    count = 0
    for i in range(0, 250):
        # temp=dataset_train.X[i*2499:(i+1)*2499,:]
        tensorX_train[:, count, :] = dataset_train.X[i * 3000:(i + 1) * 3000, :]
        tensorY_train[:, count, :] = dataset_train.y[i * 3000:(i + 1) * 3000, :]
        count = count + 1

    tensorX_train = np.nan_to_num(tensorX_train).astype(np.float32)
    tensorY_train = np.nan_to_num(tensorY_train).astype(np.float32)
    tensorX_train = torch.from_numpy(tensorX_train)
    tensorY_train = torch.from_numpy(tensorY_train)
    return tensorX_train, tensorY_train

def generate_tensor_validation(dataset_train):
    tensorX_val = np.full((3000, 45, 6), np.nan)
    tensorY_val = np.full((3000, 45, 6), np.nan)
    count = 0
    for i in range(0, 45):
        # temp=dataset_train.X[i*2499:(i+1)*2499,:]
        tensorX_val[:, count, :] = dataset_train.X[i * 3000:(i + 1) * 3000, :]
        tensorY_val[:, count, :] = dataset_train.y[i * 3000:(i + 1) * 3000, :]
        count = count + 1

    tensorX_val = np.nan_to_num(tensorX_val).astype(np.float32)
    tensorY_val = np.nan_to_num(tensorY_val).astype(np.float32)
    tensorX_val = torch.from_numpy(tensorX_val)
    tensorY_val = torch.from_numpy(tensorY_val)
    return tensorX_val, tensorY_val


def generate_tensor_test(dataset_test):
    tensorX_test = np.full((3000, 5, 6), np.nan)
    tensorY_test = np.full((3000, 5, 6), np.nan)

    count = 0
    for i in range(0, 5):
        # temp=dataset_train.X[i*2499:(i+1)*2499,:]
        tensorX_test[:, count, :] = dataset_test.X[i * 3000:(i + 1) * 3000, :]
        tensorY_test[:, count, :] = dataset_test.y[i * 3000:(i + 1) * 3000, :]
        count = count + 1

    tensorX_test = np.nan_to_num(tensorX_test).astype(np.float32)
    tensorY_test = np.nan_to_num(tensorY_test).astype(np.float32)
    tensorX_test = torch.from_numpy(tensorX_test)
    tensorY_test = torch.from_numpy(tensorY_test)
    return tensorX_test, tensorY_test


train_X, train_Y = generate_tensor_train(train)
test_X, test_Y = generate_tensor_test(test)
validation_X, validation_Y =generate_tensor_validation(validation)
train_sample_weights = 1 - np.isnan(train_X[:, :, 0])
val_sample_weights = 1 - np.isnan(validation_X[:, :, 0])

class RNNLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(RNNLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


####Definition of the feed-forward-neural network###
class TNNCell(jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.ts = 5e-4
        self.net = nn.Sequential(nn.Linear(12, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 6),
                                 )

    def forward(self, inp: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        prev_out = hidden
        combined = torch.cat((inp, hidden), dim=1)
        out = hidden + self.ts * self.net(combined)  # f_[k+1]=f[k]+ts*delta_f(P)
        return hidden, out


rnn = RNNLayer(TNNCell)
#rnn.load_state_dict(torch.load("C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/RNN_SSNN_first_solution.pth"))
rnn.load_state_dict(torch.load("C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/SSNN_saved_model_HPO_1.pth"))


###Implementattion of the Test Dataset######
with torch.no_grad():
    out_test, hidden_test = rnn(test_X[:, :, :], test_Y[0, :, :])

list_quantities=['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']
list_all_predic=[]
list_values=[]
y=0


lower_limit=885000
upper_limit=887999
load_profile=0


plt.rcParams.update({'font.size': 12})
y_pred=out_test[:,load_profile,0] * (df_all_data_before_trafo_output['master.freq'].max() - df_all_data_before_trafo_output[
        'master.freq'].min()) + \
        df_all_data_before_trafo_output['master.freq'].min()
y_pred=pd.DataFrame(y_pred.numpy())
y_real=df_test_before_trafo_output['master.freq'].loc[lower_limit:upper_limit].reset_index(drop=True)
plt.title('Master-Frequenz')
y_pred=y_pred.to_numpy().reshape(-1)
y_real=y_real.to_numpy()
plt.plot(np.arange(0, 1.500, 5e-4), y_real, label= r'$f_\mathrm{real}$')
plt.plot(np.arange(0, 1.500, 5e-4), y_pred, label=r'$^f_\mathrm{pred}$')
plt.xlabel('$t\,/\,\mathrm{s}$')
plt.ylabel(r'$f_\mathrm{Master}\,/\,\mathrm{Hz}$')
plt.text(x=0.1, y=51.5,s=f'MSE: {((y_pred-y_real)**2).sum() / len(y_real):.4f}'+' $\mathrm{Hz}^2$')
plt.legend()
plt.ylim(48,52)
plt.show()

y_pred=out_test[:,load_profile,1] * (df_all_data_before_trafo_output['slave.freq'].max() - df_all_data_before_trafo_output[
        'slave.freq'].min()) + \
        df_all_data_before_trafo_output['slave.freq'].min()
y_pred=pd.DataFrame(y_pred.numpy())
y_real=df_test_before_trafo_output['slave.freq'].loc[lower_limit:upper_limit].reset_index(drop=True)
plt.title('Slave-Frequenz')
y_pred=y_pred.to_numpy().reshape(-1)
y_real=y_real.to_numpy()
plt.plot(np.arange(0, 1.500, 5e-4), y_real, label= r'$f_\mathrm{real}$')
plt.plot(np.arange(0, 1.500, 5e-4), y_pred, label=r'$^f_\mathrm{pred}$')
plt.xlabel('$t\,/\,\mathrm{s}$')
plt.ylabel(r'$f_\mathrm{Slave}\,/\,\mathrm{Hz}$')
plt.text(x=0.1, y=51.5,s=f'MSE: {((y_pred-y_real)**2).sum() / len(y_real):.4f}'+' $\mathrm{Hz}^2$')
plt.legend()
plt.ylim(48,52)
plt.show()

y_pred=out_test[:,load_profile,2] * (df_all_data_before_trafo_output['load.freq'].max() - df_all_data_before_trafo_output[
        'load.freq'].min()) + \
        df_all_data_before_trafo_output['load.freq'].min()
y_pred=pd.DataFrame(y_pred.numpy())
y_real=df_test_before_trafo_output['load.freq'].loc[lower_limit:upper_limit].reset_index(drop=True)
plt.title('Last-Frequenz')
y_pred=y_pred.to_numpy().reshape(-1)
y_real=y_real.to_numpy()
plt.plot(np.arange(0, 1.500, 5e-4), y_real, label= r'$f_\mathrm{real}$')
plt.plot(np.arange(0, 1.500, 5e-4), y_pred, label=r'$^f_\mathrm{pred}$')
plt.xlabel('$t\,/\,\mathrm{s}$')
plt.ylabel(r'$f_\mathrm{Load}\,/\,\mathrm{Hz}$')
plt.text(x=0.1, y=51.5,s=f'MSE: {((y_pred-y_real)**2).sum() / len(y_real):.4f}'+' $\mathrm{Hz}^2$')
plt.legend()
plt.ylim(48,52)
plt.show()

y_pred=out_test[:,load_profile,3] * (df_all_data_before_trafo_output['master.voltage'].max() - df_all_data_before_trafo_output[
        'master.voltage'].min()) + \
        df_all_data_before_trafo_output['master.voltage'].min()
y_pred=pd.DataFrame(y_pred.numpy())
y_real=df_test_before_trafo_output['master.voltage'].loc[lower_limit:upper_limit].reset_index(drop=True)
plt.title('Master-Spannung')
y_pred=y_pred.to_numpy().reshape(-1)
y_real=y_real.to_numpy()
plt.plot(np.arange(0, 1.500, 5e-4), y_real, label= r'$U_\mathrm{real}$')
plt.plot(np.arange(0, 1.500, 5e-4), y_pred, label=r'$^U_\mathrm{pred}$')
plt.xlabel('$t\,/\,\mathrm{s}$')
plt.ylabel(r'$U_\mathrm{Master}\,/\,\mathrm{V}$')
plt.text(x=0.11, y=100,s=f'MSE: {((y_pred-y_real)**2).sum() / len(y_real):.2f}'+' $\mathrm{V}^2$')
plt.legend()
plt.show()

y_pred=out_test[:,load_profile,4] * (df_all_data_before_trafo_output['slave.voltage'].max() - df_all_data_before_trafo_output[
        'slave.voltage'].min()) + \
        df_all_data_before_trafo_output['slave.voltage'].min()
y_pred=pd.DataFrame(y_pred.numpy())
y_real=df_test_before_trafo_output['slave.voltage'].loc[lower_limit:upper_limit].reset_index(drop=True)
plt.title('Slave-Spannung')
y_pred=y_pred.to_numpy().reshape(-1)
y_real=y_real.to_numpy()
plt.plot(np.arange(0, 1.500, 5e-4), y_real, label= r'$U_\mathrm{real}$')
plt.plot(np.arange(0, 1.500, 5e-4), y_pred, label=r'$^U_\mathrm{pred}$' )
plt.xlabel('$t\,/\,\mathrm{s}$')
plt.ylabel(r'$U_\mathrm{Slave}\,/\,\mathrm{V}$')
plt.text(x=0.11, y=100,s=f'MSE: {((y_pred-y_real)**2).sum() / len(y_real):.2f}'+' $\mathrm{V}^2$')
plt.legend()
plt.show()

y_pred=out_test[:,load_profile,5] * (df_all_data_before_trafo_output['load.voltage'].max() - df_all_data_before_trafo_output[
        'load.voltage'].min()) + \
        df_all_data_before_trafo_output['load.voltage'].min()
y_pred=pd.DataFrame(y_pred.numpy())
y_real=df_test_before_trafo_output['load.voltage'].loc[lower_limit:upper_limit].reset_index(drop=True)
plt.title('Last-Spannung')
y_pred=y_pred.to_numpy().reshape(-1)
y_real=y_real.to_numpy()
plt.plot(np.arange(0, 1.500, 5e-4), y_real, label= r'$U_\mathrm{real}$')
plt.plot(np.arange(0, 1.500, 5e-4), y_pred, label=r'$^U_\mathrm{pred}$')
plt.xlabel('$t\,/\,\mathrm{s}$')
plt.ylabel(r'$U_\mathrm{Load}\,/\,\mathrm{V}$')
plt.text(x=0.11, y=100,s=f'MSE: {((y_pred-y_real)**2).sum() / len(y_real):.2f}'+' $\mathrm{V}^2$')
plt.legend()
plt.show()

x=1
