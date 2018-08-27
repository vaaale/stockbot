
import matplotlib.pyplot as plt
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
        y = 0
#         y = np.sin(2*np.pi*x*f) * 1
        y = y + np.random.uniform(-0.5, 0.5, timesteps).cumsum()
        data.append(y)
    data = np.asarray(data)
    return data



def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


prices = generate_data()
prices = prices[0,:]
rsi = rsiFunc(prices)

print('Done')