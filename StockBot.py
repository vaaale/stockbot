#https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf

import gym
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from game import Game

env = Game()
# torch.manual_seed(1)

# Hyperparameters
learning_rate = 0.01
gamma = 0.8


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(-1)
        return x


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space
        self.action_space = env.action_space.n
        self.nb_features = env.nb_features

        # Define model
        # self.conv1 = nn.Conv1d(self.nb_features, 64, 5, bias=False)
        # self.conv2 = nn.Conv1d(64, 128, 5, bias=False)
        # self.fc1 = nn.Linear(128*(env.lookback_window-8), self.action_space, bias=False)
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = []
        self.np_policy_history = []
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        # model = torch.nn.Sequential(
        #     self.l1,
        #     nn.Dropout(p=0.6),
        #     nn.ReLU(),
        #     self.l2
        # )
        # logits = model(x)
        l1_out = self.l1(x)
        d_out = nn.Dropout(p=0.6)(l1_out)
        l2_out = self.l2(d_out)
        rel_out = nn.ReLU()(l2_out)
        probs = self.softmax(rel_out)
        return probs, l1_out, d_out, l2_out, rel_out


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def select_action(ob):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state_history = torch.from_numpy(ob[0].reshape(policy.state_space)).float()
    probs, l1_out, d_out, l2_out, rel_out = policy(state_history)
    c = Categorical(probs)
    action = c.sample()

    # Add log probability of our chosen action to our history
    policy.policy_history.append(c.log_prob(action))
    policy.np_policy_history.append(c.log_prob(action).detach())
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

def ann_update_policy():
    R = 0
    rewards = []

    # print('REWARD HISTORY')
    # print(policy.reward_episode)
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    rewards = np.asarray(rewards, dtype=np.float32)

    # Scale rewards
    t_rewards = torch.FloatTensor(rewards)
    eps = float(np.finfo(np.float32).eps)
    print('t_mean {}'.format(t_rewards.mean()))
    print('t_std {}'.format(t_rewards.std()))

    print('np_mean {}'.format(np.mean(rewards)))
    print('np_std {}'.format(np.std(rewards, ddof=1)))
    t_rewards = (t_rewards - t_rewards.mean()) / (t_rewards.std() + eps)
    np_rewards = (rewards - np.mean(rewards)) / (np.std(rewards, ddof=1) + eps)

    print('t_rewards {}'.format(t_rewards))
    print('np_rewards {}'.format(np_rewards))

    # Calculate loss
    t_history = torch.stack(policy.policy_history, 0)
    np_history = np.hstack(policy.np_policy_history)
    print('t_history {}'.format(t_history))
    print('np_history {}'.format(np_history))

    # loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))
    hist_reward = torch.mul(t_history, t_rewards)
    np_hist_reward = np_history * np_rewards
    print('hist_reward {}'.format(hist_reward))
    print('np_hist_reward {}'.format(np_hist_reward))

    loss = torch.sum(hist_reward.mul(-1), -1)
    np_loss = np.sum((np_hist_reward*-1), -1)
    print('t_loss {}'.format(loss))
    print('np_loss {}'.format(np_loss))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = []
    policy.np_policy_history = []
    policy.reward_episode = []


def plot_stats(episode):
    data = env.close
    xs = np.arange(len(data))
    x_byes = env.byes
    x_sells = env.sells

    print('Byes: {}'.format(x_byes))
    print('Sells: {}'.format(x_sells))
    print('Nothing: {}'.format(env.nothing))

    print('Miss Byes: {}'.format(env.miss_byes))
    print('Miss Sells: {}'.format(env.miss_sells))


    y_byes = [data[b] for b in x_byes]
    y_sells = [data[s] for s in x_sells]

    plt.plot(xs, data)
    plt.scatter(x_byes, y_byes, color='green', marker='o')
    plt.scatter(x_sells, y_sells, color='blue', marker='X')
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
            print('Result funds: {}'.format(env.funds))
            plot_stats(episode)

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,
                                                                                                        time))
            break


episodes = 1000
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