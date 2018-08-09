import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from game import Game
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

num_periods = 15
lookback_window = 10
state_size = 50
num_actions = 3

batch_size = 32

random.seed(111)


def generate_batch(batch_size, num_periods):
    x_batches = np.random.uniform(-10, 10, size=[batch_size, num_periods*lookback_window]).cumsum(axis=1).reshape([-1, 1, num_periods*lookback_window])
    return x_batches


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 5)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(16*2, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 2)
        x = F.sigmoid(self.fc1(x))
        return x

net = Net()
print(net)

env = Game(batch_size, num_periods, lookback_window)
ob = env.reset()

obs, acs, rews = [], [], []
done = False
while not done:
    x = torch.from_numpy(ob).float()
    with torch.no_grad():
        obs.append(ob)
        ac = net.forward(x)
    acs.append(ac)
    ob, rew, done = env.step(ob, ac)
    rews.append(rew)

# Compute Q-Values

for reward in rews:
    print('Test')