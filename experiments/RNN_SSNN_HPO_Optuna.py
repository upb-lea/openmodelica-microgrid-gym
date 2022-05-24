import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
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
from pytorchtools_hand_written import EarlyStopping
import importlib

import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

# vergangenheitswerte noch mit reingeben
np.set_printoptions(precision=20)
import pickle5 as pickle

save_path1 = "C:/Users/Marvin/OneDrive/Masterarbeit/openmodelica-microgrid-gym/experiments/swing_equation/net_train_P_f1_exp_moreData.pth"
save_path = [save_path1]
ts = 5e-4


###########Restructuring of the data####################################################################################
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


df_all_data = pd.read_pickle('Microgrid_Data_stochastic_component')

list_columns = ['master.freq', 'master.instPow', 'master.instQ', 'master.voltage', 'slave.freq', 'slave.instPow',
                'slave.instQ', 'slave.voltage', 'load.freq', 'Load.activePower', 'load.reactivePower', 'load.voltage']
df_validation_prep = pd.read_pickle('Microgrid_Data_validation')


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
df_all_data_before_trafo_output = df_all_data[
    ['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]

df_training = df_all_data.iloc[:750000, :]
df_validation = df_all_data.iloc[750000:885000, :]
df_test = df_all_data.iloc[885000:, :]

# perform z-transformation --> can be shorted later, when the real frequency on the load bus is used
df_training_before_trafo_output = df_training[
    ['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
df_test_before_trafo_output = df_test[
    ['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
df_test_before_trafo_input = df_test[
    ['master.instPow', 'slave.instPow', 'Load.activePower', 'master.instQ', 'slave.instQ', 'load.reactivePower']]
df_validation_before_trafo_output = df_validation[
    ['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
df_validation_before_trafo_input = df_validation[
    ['master.instPow', 'slave.instPow', 'Load.activePower', 'master.instQ', 'slave.instQ', 'load.reactivePower']]


def df_min_max_transformation(df):
    for columns2 in list_columns:
        df[columns2] = (df[columns2] - df[columns2].min()) / (df[columns2].max() - df[columns2].min())
    return df


df_all_data = df_min_max_transformation(df_all_data)
df_training = df_all_data.iloc[:750000, :]
df_validation = df_all_data.iloc[750000:885000, :]
df_test = df_all_data.iloc[885000:, :]
df_test_output = df_test[['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]


####################Restructuring of the ANN 2.0########################################################################

class CSVDataset:
    # load the dataset
    def __init__(self, df_in, df_out):
        # store the inputs and outputs
        self.X = df_in.to_numpy()  # one dimensional input: power
        self.y = df_out.to_numpy()

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

df_validation_input = df_validation[
    ['master.instPow', 'slave.instPow', 'Load.activePower', 'master.instQ', 'slave.instQ', 'load.reactivePower']]
df_validation_output = df_validation[
    ['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
validation = CSVDataset(df_validation_input, df_validation_output)

df_test_input = df_test[
    ['master.instPow', 'slave.instPow', 'Load.activePower', 'master.instQ', 'slave.instQ', 'load.reactivePower']]
df_test_output = df_test[['master.freq', 'slave.freq', 'load.freq', 'master.voltage', 'slave.voltage', 'load.voltage']]
test = CSVDataset(df_test_input, df_test_output)


##########Change into tensors###########################################################################################
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
validation_X, validation_Y = generate_tensor_validation(validation)
train_sample_weights = 1 - np.isnan(train_X[:, :, 0])
val_sample_weights = 1 - np.isnan(validation_X[:, :, 0])


######################################OPTUNA functions for Hyperparameteroptimization###################################
def define_model(trial):
    # We optimize the number of layers and hidden units in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    activation_func = trial.suggest_categorical("activ_func", ["Sigmoid", "tanh", "ReLU"])
    if activation_func == "Sigmoid":
        activation_func_module = nn.Sigmoid()
    if activation_func == "tanh":
        activation_func_module = nn.Tanh()
    if activation_func == "ReLU":
        activation_func_module = nn.ReLU()
    x = 1
    in_features = 12
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 64, 150)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation_func_module)
        in_features = out_features
    layers.append(nn.Linear(in_features, 6))

    return nn.Sequential(*layers)


#########Definition of the SSNN#########################################################################################
class RNNLayer(jit.ScriptModule):
    def __init__(self, cell, trial, *cell_args):
        super(RNNLayer, self).__init__()
        self.cell = cell(trial, *cell_args)

    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


####Definition of the SSNN cell#########################################################################################
class TNNCell(jit.ScriptModule):
    def __init__(self, trial):
        super().__init__()
        self.ts = 5e-4
        self.net = define_model(trial)

    def forward(self, inp: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        prev_out = hidden
        combined = torch.cat((inp, hidden), dim=1)
        out = hidden + self.ts * self.net(combined)  # f_[k+1]=f[k]+ts*delta_f(P)
        return hidden, out


loss_func = nn.MSELoss(reduction="none")

val_epoch_loss = 0
loss_stats = {
    'train': [],
    "val": []
}

n_epochs = 200
list_loss_train = []
list_loss_val = []
check_list = [0, 0, 0]


def objective(trial):
    early_stopping = EarlyStopping(patience=35, verbose=True, path='Gridsearch{}.pth'.format(str(trial.number)))
    # Generate the rnn.
    rnn = RNNLayer(TNNCell, trial)
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adamax", "SGD"])
    lr = trial.suggest_float("lr", 0.0008, 0.0013, log=True)  # change it
    tbptt_size = trial.suggest_int("tbptt_size", 10, 250, log=True)
    optimizer = getattr(optim, optimizer_name)(rnn.parameters(), lr=lr)

    with tqdm(desc="Training", total=n_epochs) as pbar:
        n_batches_train = np.ceil(train_X.shape[0] / tbptt_size).astype(int)
        n_batches_val = np.ceil(test_X.shape[0] / tbptt_size).astype(int)
        for epoch in range(n_epochs):
            tracker = 0
            train_epoch_loss = 0
            hidden = train_Y[0, :, :]

            for i in range(n_batches_train):
                rnn.zero_grad()
                output, hidden = rnn(train_X[i * tbptt_size:(i + 1) * tbptt_size, :, :], hidden.detach())
                loss = loss_func(output, train_Y[i * tbptt_size:(i + 1) * tbptt_size, :, :])
                loss = (loss * train_sample_weights[i * tbptt_size:(i + 1) * tbptt_size, :,
                               None] / train_sample_weights[
                                       i * tbptt_size:(
                                                              i + 1) * tbptt_size,
                                       :].sum()).sum().mean()
                loss.backward()
                optimizer.step()

            pbar.update()
            pbar.set_postfix_str(f'train_loss: {loss.item():.4f}')
            list_loss_train.append(loss)
            with torch.no_grad():
                tracker = 0
                test_epoch_loss = 0
                rnn.eval()
                hidden = validation_Y[0, :, :]
                for i in range(n_batches_val):
                    output, hidden = rnn(validation_X[i * tbptt_size:(i + 1) * tbptt_size, :, :], hidden.detach())
                    loss_val = loss_func(output, validation_Y[i * tbptt_size:(i + 1) * tbptt_size, :, :])
                    loss_val = (loss_val * val_sample_weights[i * tbptt_size:(i + 1) * tbptt_size, :,
                                           None] / val_sample_weights[
                                                   i * tbptt_size:(
                                                                          i + 1) * tbptt_size,
                                                   :].sum()).sum().mean()

                list_loss_val.append(loss_val)
                early_stopping(loss_val, rnn)

                if early_stopping.early_stop or np.isnan(loss_val.item()):
                    print("Early stopping")
                    loss_val = early_stopping.val_loss_min
                    break

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss_val.item()


TPE_sampler = TPESampler(
    n_startup_trials=20)  ###check TPE sampler in documentation#####################################

study = optuna.create_study(direction="minimize", study_name="Second_try", storage=f'sqlite:///optuna.sqlite',
                            sampler=TPE_sampler)
study.optimize(objective, n_trials=100, timeout=None)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


#######################Print of the HPO result##########################################################################
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Number: ", trial.number)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
x = 1
