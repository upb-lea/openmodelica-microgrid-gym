from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset, DataLoader
import torch
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

writer = SummaryWriter()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(21, 320)
        # self.bn1 = nn.BatchNorm1d(320)
        self.fc2 = nn.Linear(320, 320)
        self.fc22 = nn.Linear(320, 320)
        #self.fc23 = nn.Linear(320, 320)
        #self.fc24 = nn.Linear(320, 320)
        # self.bn2 = nn.BatchNorm1d(320)
        self.fc3 = nn.Linear(320, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc22(x))
        #x = F.relu(self.fc23(x))
        #x = F.relu(self.fc24(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    x = Tensor(np.load('data-gen_fully-random/X.npy'))  # transform to torch tensor
    y = Tensor(np.load('data-gen_fully-random/y.npy'))

    cut = int(x.size(0) * .80)
    train = TensorDataset(x[:cut], y[:cut])  # create your datset
    test = TensorDataset(x[cut:], y[cut:])  # create your datset
    trainloader = DataLoader(train, batch_size=512, shuffle=True, num_workers=2)
    testloader = DataLoader(test, batch_size=4, shuffle=False, num_workers=2)

    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
    n = 0
    loss = np.infty
    for epoch in trange(2):  # loop over the dataset multiple times
        t = tqdm(trainloader, mininterval=1, leave=False)
        for data in t:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            n += 1
            if n % 500 == 0:
                writer.add_scalar('Loss/train', loss.item(), n)
                # writer.add_scalar('Loss/test', np.random.random(), n_iter)

            t.set_postfix(loss=loss.item(), refresh=False)
        if epoch % 49 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, f'snapshots/{datetime.now().isoformat()}lrdecay.995_data.05')

        scheduler.step()

    print('Finished Training')

    with torch.no_grad():
        cum_loss = 0
        for data in testloader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            cum_loss += loss.item()

    print(cum_loss / len(testloader))
