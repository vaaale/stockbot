
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

    def __init__(self, batch_size, num_periods):
        self.states = np.full(batch_size, self.ST_NOT_INVESTED)
        self.rewards = np.zeros(shape=(batch_size, num_periods), dtype=np.float32)
        self.observations = []

    def _sell(self, row, _ob):
        reward = 0
        _state = self.states[row]

        if self.ST_INVESTED == _state:
            reward += (_ob - self.observations[row])
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
        for row, ob, ac in zip(range(len(observations)), observations, actions):
            # Update states
            if self.AC_SELL == ac:
                self._sell(row, ob)
            elif self.AC_INVEST == ac:
                self._invest(row, ob)
            elif self.AC_NOTHING == ac:
                self._nothing(row, ob)

            self.observations.append(ob)

        return self.rewards

    def play(self, observations, actions):
        batch_size, num_periods, _ = observations.shape
        for si in range(batch_size):
            for ti in range(num_periods):
                ob = observations[si][ti][0]
                ac = actions[si][ti]

                # Update states
                if self.AC_SELL == ac:
                    self.rewards[si][ti] += self._sell(si, ob)
                elif self.AC_INVEST == ac:
                    self.rewards[si][ti] += self._invest(si, ob)
                elif self.AC_NOTHING == ac:
                    self.rewards[si][ti] += self._nothing(si, ob)

                self.observations.append(ob)

        return self.rewards
