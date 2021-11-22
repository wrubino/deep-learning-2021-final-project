import numpy as np
import matplotlib.pyplot as plt

fs = 44000
f = 100

t = np.arange(0, 0.05, 1 / fs)

data_1 = np.cos(2 * np.pi * f * t)

# data_1 += np.random.uniform(size=data_1.shape)   # noise

data_2 = np.cos(2 * np.pi * f * (t - 0.003))

# data_2 += np.random.uniform(size=data_2.shape)   # noise
lags, c, _, _ = plt.xcorr(data_1, data_2, maxlags=int(fs * 0.01))

lag_samples =
lag = lags[np.argmax(c)] / fs

plt.show()
plt.plot(t, data_1, 'r')
plt.plot(t, data_2, 'b')
plt.show()
