
import numpy as np
import pandas as pd

nb_sources = 10
f = 5
timesteps = 1000
lookback_window = 30


def generate_data():
    data = []
    for i in range(nb_sources):
        phase = np.random.uniform(-0.5, 0.5, 1)
#         f = np.random.uniform(1, 50, 1)
        x = np.linspace(0, 1, timesteps) + phase
        y = np.sin(2*np.pi*x*f) * 1
        y = y + np.random.uniform(-0.5, 0.5, timesteps).cumsum()
        data.append(y)
    data = np.asarray(data)
    return data


def generate_features(data):
    df = pd.DataFrame(data)
    df = df.T
    df_pct = ((df.shift(-1) - df) / df.abs())
    df_pct = np.asarray([df_pct[i:i + lookback_window].T.values for i in range(0, df_pct.shape[0] - lookback_window, 1)])
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
        self.lookback_window = lookback_window
        self.states = np.full(nb_sources, self.ST_NOT_INVESTED)
        self.rewards = np.zeros(shape=(nb_sources), dtype=np.float32)
        self.observations = []
        pct_data = generate_features(generate_data())
        self.data = pct_data
        self.num_periods = len(self.data)
        # self.data = np.random.uniform(-10, 10, size=[batch_size, num_periods*lookback_window]).cumsum(axis=1).reshape([-1, 1, num_periods*lookback_window])
        self._step = 0


    def reset(self):
        batch =  self.data[self._step]
        batch = batch.reshape(nb_sources, 1, lookback_window)
        self._step += 1
        return batch


    def _sell(self, row, _ob):
        reward = 0
        _state = self.states[row]

        if self.ST_INVESTED == _state:
            reward += (_ob.reshape(self.lookback_window)[-1] - self.observations[row].reshape(self.lookback_window)[-1])
            self.states[row] = self.ST_NOT_INVESTED
        else:
            reward += 0  # One could consider rewarding the fact that this is a correct decision IF we where invested

        return reward

    def _invest(self, row, _ob):
        reward = 0
        _state = self.states[row]

        if self.ST_NOT_INVESTED == _state:
            reward += 0  # Because we dont know wether it is a good action or not.
            self.states[row] = self.ST_INVESTED

        return reward

    def _nothing(self, row, _ob):
        # Might give com creds for doing nothing in some circumstances
        reward = 0
        return reward

    def step(self, observations, action):
        rewards = np.zeros(shape=(nb_sources), dtype=np.float32)
        for row, ob, ac in zip(range(len(observations)), observations, action):
            # Update states
            if self.AC_SELL == ac:
                sell = self._sell(row, ob)
                rewards[row] = sell
            elif self.AC_INVEST == ac:
                rewards[row] = self._invest(row, ob)
            elif self.AC_NOTHING == ac:
                rewards[row] = self._nothing(row, ob)

            self.observations.append(ob)

        batch = self.data[self._step]
        batch = batch.reshape(nb_sources, 1, lookback_window)
        self._step += 1
        done = self._step == self.num_periods-1

        return batch, rewards, done

    def play(self, observations, actions):
        action_ind = np.argmax(actions, axis=1)
        batch_size, num_periods, _ = observations.shape
        for si in range(batch_size):
            ob = observations[si][0][-1]
            ac = action_ind[si]

            # Update states
            if self.AC_SELL == ac:
                self.rewards[si] += self._sell(si, ob)
            elif self.AC_INVEST == ac:
                self.rewards[si] += self._invest(si, ob)
            elif self.AC_NOTHING == ac:
                self.rewards[si] += self._nothing(si, ob)

            self.observations.append(ob)

        return self.rewards
