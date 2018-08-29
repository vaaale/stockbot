
import numpy as np
import pandas as pd
from gym import spaces
from gym.envs.registration import EnvSpec

f = 5
timesteps = 100
lookback_window = 30


def generate_data():
    phase = np.random.uniform(-0.5, 0.5, 1)
#         f = np.random.uniform(1, 50, 1)
    time = np.linspace(0, 1, timesteps)
    x = np.sin(2*np.pi*(time + phase) * f) * 1
    # x = x + np.random.uniform(-0.5, 0.5, timesteps).cumsum()
    data = np.asarray(x)
    return data


def generate_features(data):
    df = pd.DataFrame(data)
    df_pct = ((df.shift(-1) - df) / df.abs())
    return df_pct, data


def generate_batches():
    data = generate_data()
    # data = np.arange(0,100, 1)
    features, data = generate_features(data)
    # features = pd.DataFrame(data)

    batches = np.asarray([features[i:i + lookback_window].values for i in range(0, features.shape[0] - lookback_window, 1)])
    return batches, data

class Game:
    ST_NOT_INVESTED = 0
    ST_INVESTED = 1

    AC_BUY = 0
    AC_SELL = 1
    AC_NOTHING = 2

    START_CAPITAL = 0

    def __init__(self):
        self.observation_space = lookback_window
        self.action_space = spaces.Discrete(2)
        self.lookback_window = lookback_window
        self.state = self.ST_NOT_INVESTED
        self.price = 0
        self.rewards = 0
        self.data, self.raw_data = generate_batches()
        self.num_periods = len(self.data)
        self._step = 0
        self.spec = EnvSpec(id='StockBroker-v1', reward_threshold=1000)
        self.done = False
        self.byes = []
        self.sells = []
        self.funds = self.START_CAPITAL

    def _sell(self, _ob):
        reward = 0
        if self.ST_INVESTED == self.state:
            # rew = (self.price - _ob.reshape(-1)[-1])
            current_price = self.raw_data[self._step+lookback_window-1]
            rew = current_price - self.price

            if current_price != _ob.reshape(-1)[-1]:
                print("Sell Prices differ: {} vs {}".format(self.price , _ob.reshape(-1)[-1]))
                print('')



            self.funds += rew
            # print(self.funds)
            self.state = self.ST_NOT_INVESTED
            self.price = 0
            self.sells.append(self._step + lookback_window)
            reward = 1

        return reward

    def _invest(self, _ob):
        reward = 1

        if self.ST_NOT_INVESTED == self.state:
            self.state = self.ST_INVESTED
            # self.price = _ob.reshape(-1)[-1]
            self.price = self.raw_data[self._step+lookback_window-1]
            if self.price != _ob.reshape(-1)[-1]:
                print("Bye Prices differ: {} vs {}".format(self.price , _ob.reshape(-1)[-1]))
                print('')

            self.byes.append(self._step + lookback_window)

        return reward

    def _nothing(self, _ob):
        # Might give com creds for doing nothing in some circumstances
        reward = 1
        return reward

    def _next(self):
        batch = self.data[self._step]
        batch = batch.reshape(-1, 1, 30)
        # state_feature = np.empty_like(batch)
        # state_feature.fill(self.state)
        # batch = np.hstack([batch, state_feature])
        return batch

    def reset(self):
        self.data, self.raw_data = generate_batches()
        self._step = 0
        self.price = 0
        self.rewards = 0
        self.state = self.ST_NOT_INVESTED
        self.funds = self.START_CAPITAL

        self.done = False
        batch = self._next()
        self.byes = []
        self.sells = []


        return batch

    def step(self, observations, action):
        reward = 0
        if self.AC_BUY == action:
            reward = self._invest(observations)
        elif self.AC_SELL == action:
            sell = self._sell(observations)
            reward = sell
        elif self.AC_NOTHING == action:
            reward = self._nothing(observations)

        if self.funds < 0:
            # print('Busted!')
            self.done = True
            reward = self.funds
            print("self.funds: {}, reward: {}".format(self.funds, reward))
        else:
            reward = 0

        self._step += 1
        done = (self._step == self.num_periods-1) or self.done
        batch = self._next()

        return batch, reward, done, dict()
