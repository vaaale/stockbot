import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from torch.distributions import Categorical

from game import Game
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

random.seed(111)



#
# data = generate_data()
# data_pct = generate_features(data)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 5)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(16*22, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 22)
        x = F.sigmoid(self.fc1(x))
        return x

net = Net()
print(net)

env = Game()
ob = env.reset()

obs, acs, rews = [], [], []
done = False
while not done:

    x = torch.from_numpy(ob).float()
    with torch.no_grad():
        obs.append(ob)
        probs = net.forward(x)
        m = Categorical(probs)
        ac = m.sample()

    acs.append(ac.numpy())
    ob, rew, done = env.step(ob, ac.numpy())
    loss = -m.log_prob(ac) * torch.from_numpy(rew).float()
    rews.append(rew)

# Compute Q-Values
gamma = 1
q_n = []
reward_to_go = False
for path in rews:
    q = 0
    q_path = []

    # Dynamic programming over reversed path
    for rew in reversed(path):
        q = rew + gamma * q
        q_path.append(q)
    q_path.reverse()

    # Append these q values
    if not reward_to_go:
        q_path = [q_path[0]] * len(q_path)
    q_n.extend(q_path)



optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()

loss.backward()
optimizer.step()
