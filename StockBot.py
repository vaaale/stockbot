#https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

# env = gym.make('CartPole-v1')
from game import Game

env = Game()
# torch.manual_seed(1)

# Hyperparameters
learning_rate = 0.01
gamma = 0.99


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(-1)
        return x


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space
        self.action_space = env.action_space.n

        # Define model
        # self.l1 = nn.Linear(self.state_space, 128, bias=False)
        # self.l2 = nn.Linear(128, self.action_space, bias=False)
        self.conv1 = nn.Conv1d(1, 64, 5)
        self.conv2 = nn.Conv1d(64, 128, 5)
        self.fc1 = nn.Linear(128*22, self.action_space)

        self.gamma = gamma

        # Episode policy and reward history
        # self.policy_history = Variable(torch.Tensor())
        self.policy_history = []
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        # model = torch.nn.Sequential(
        #     self.l1,
        #     nn.Dropout(p=0.6),
        #     nn.ReLU(),
        #     self.l2,
        #     nn.Softmax(dim=-1)
        # )
        # return model(x)
        model = torch.nn.Sequential(
            self.conv1,
            nn.Tanh(),
            self.conv2,
            nn.Tanh(),
            Flatten(),
            self.fc1,
            nn.Softmax(dim=-1),
            Flatten()
        )
        y = model(x)

        return y


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def select_action(ob):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state1 = torch.from_numpy(ob).float()
    state2 = policy(Variable(state1))
    c = Categorical(state2)
    action = c.sample()

    # Add log probability of our chosen action to our history
    policy.policy_history.append(c.log_prob(action))
    return action


def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    eps = float(np.finfo(np.float32).eps)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    # Calculate loss
    history = torch.stack(policy.policy_history, 0)
    # loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))
    loss = (torch.sum(torch.mul(history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    # policy.policy_history = Variable(torch.Tensor())
    policy.policy_history = []
    policy.reward_episode = []


def plot_stats(episode):
    data = env.raw_data
    xs = np.arange(len(data))
    byes = env.byes
    sells = env.sells

    print('Byes: {}'.format(byes))
    print('Sells: {}'.format(sells))

    ys2 = [data[b] for b in byes]
    ys3 = [data[s] for s in sells]

    plt.plot(xs, data)
    plt.scatter(byes, ys2, color='green', marker='o')
    plt.scatter(sells, ys3, color='blue', marker='X')
    plt.savefig('figures/plot{}.png'.format(episode))
    plt.clf()


def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state
        done = False

        for time in range(1000):
            action = select_action(state)
            # Step through environment using chosen action
            # state, reward, done, _ = env.step(action.data[0])
            state, reward, done, _ = env.step(state, action.numpy())

            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        update_policy()

        if episode % 100 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))
            print("running_reward: {}, reward_threshold: {}, solved: {}".format(running_reward, env.spec.reward_threshold, running_reward > env.spec.reward_threshold))
            print('Last reward: {}'.format(reward))
            print('Result funds: {}'.format(env.funds))
            plot_stats(episode)

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,
                                                                                                        time))
            break


episodes = 5000
main(episodes)

window = int(episodes/20)

fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9,9]);
rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
std = pd.Series(policy.reward_history).rolling(window).std()
ax1.plot(rolling_mean)
ax1.fill_between(range(len(policy.reward_history)),rolling_mean-std, rolling_mean+std, color='orange', alpha=0.2)
ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
ax1.set_xlabel('Episode'); ax1.set_ylabel('Episode Length')

ax2.plot(policy.reward_history)
ax2.set_title('Episode Length')
ax2.set_xlabel('Episode'); ax2.set_ylabel('Episode Length')

fig.tight_layout(pad=2)
plt.show()
#fig.savefig('results.png')