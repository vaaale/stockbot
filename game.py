
import numpy as np

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

    def __init__(self, batch_size, num_periods, lookback_window):
        self.batch_size = batch_size
        self.num_periods = num_periods
        self.lookback_window = lookback_window
        self.states = np.full(batch_size, self.ST_NOT_INVESTED)
        self.rewards = np.zeros(shape=(batch_size), dtype=np.float32)
        self.observations = []
        self.data = np.random.uniform(-10, 10, size=[batch_size, num_periods*lookback_window]).cumsum(axis=1).reshape([-1, 1, num_periods*lookback_window])
        self._step = 0


    def reset(self):
        batch =  self.data[:,:,self._step*self.lookback_window:self._step*self.lookback_window+self.lookback_window]
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

    def step(self, observations, actions):
        rewards = np.zeros(shape=(self.batch_size), dtype=np.float32)
        action = np.argmax(actions.numpy(), axis=1)
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

        batch = self.data[:, :, self._step * self.lookback_window:self._step * self.lookback_window + self.lookback_window]
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
