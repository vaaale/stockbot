
import numpy as np
import pandas as pd
from gym import spaces
from gym.envs.registration import EnvSpec

f = 5
timesteps = 1000
lookback_window = 30


def generate_data():
    phase = np.random.uniform(-0.5, 0.5, 1)
#         f = np.random.uniform(1, 50, 1)
    time = np.linspace(0, 1, timesteps) + phase
    x = np.sin(2*np.pi*time*f) * 1
    x = x + np.random.uniform(-0.5, 0.5, timesteps).cumsum()
    data = np.asarray(x)
    return data


def generate_features(data):
    df = pd.DataFrame(data)
    df_pct = ((df.shift(-1) - df) / df.abs())
    df_pct = np.asarray([df_pct[i:i + lookback_window].values for i in range(0, df_pct.shape[0] - lookback_window, 1)])
    return df_pct


class Game:
    ST_NOT_INVESTED = 0
    ST_INVESTED = 1

    AC_NOTHING = 0
    AC_INVEST = 1
    AC_SELL = 2

    ALLOWED_STATES = {
        ST_NOT_INVESTED: [AC_NOTHING, AC_INVEST],
        ST_INVESTED: [AC_NOTHING, AC_SELL]
    }

    def __init__(self):
        self.observation_space = lookback_window
        self.action_space = spaces.Discrete(3)
        self.lookback_window = lookback_window
        # self.states = np.full(nb_sources, self.ST_NOT_INVESTED)
        self.states = self.ST_NOT_INVESTED
        self.price = 0
        # self.rewards = np.zeros(shape=(nb_sources), dtype=np.float32)
        self.rewards = 0
        pct_data = generate_features(generate_data())
        self.data = pct_data
        self.num_periods = len(self.data)
        # self.data = np.random.uniform(-10, 10, size=[batch_size, num_periods*lookback_window]).cumsum(axis=1).reshape([-1, 1, num_periods*lookback_window])
        self._step = 0
        self.spec = EnvSpec(id='StocBot-v1', reward_threshold=10000)

    def reset(self):
        pct_data = generate_features(generate_data())
        self.data = pct_data
        self._step = 0
        self.price = 0
        self.rewards = 0
        self.states = self.ST_NOT_INVESTED

        batch = self.data[self._step]
        # batch = batch.reshape(nb_sources, 1, lookback_window)
        self._step += 1
        return batch.reshape(-1)

    def _sell(self, _ob):
        reward = 0

        if self.ST_INVESTED == self.states:
            reward += (_ob[-1] - self.price)
            self.states = self.ST_NOT_INVESTED
            self.price = 0
        else:
            reward += 0  # One could consider rewarding the fact that this is a correct decision IF we where invested

        return reward

    def _invest(self, _ob):
        reward = 0

        if self.ST_NOT_INVESTED == self.states:
            self.states = self.ST_INVESTED
            self.price = _ob[-1]
            reward += 0  # Because we dont know wether it is a good action or not.

        return reward

    def _nothing(self, _ob):
        # Might give com creds for doing nothing in some circumstances
        reward = 0
        return reward

    def step(self, observations, action):
        reward = 0
        # rewards = np.zeros(shape=(nb_sources), dtype=np.float32)
        # for row, ob, ac in zip(range(len(observations)), observations, action):
            # Update states
        if self.AC_INVEST == action:
            reward = self._invest(observations)
        elif self.AC_SELL == action:
            sell = self._sell(observations)
            reward = sell
        elif self.AC_NOTHING == action:
            reward = self._nothing(observations)

            # self.observations.append(ob)

        batch = self.data[self._step].reshape(-1)
        self._step += 1
        done = self._step == self.num_periods

        return batch, reward, done, dict()
