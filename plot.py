import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

window = 100
data1 = pd.read_csv('./result_alpha_0.1.csv')
reward1 = convolve1d(data1['steps'] / window, np.ones(window))

data2 = pd.read_csv('./result_alpha_0.3.csv')
reward2 = convolve1d(data2['steps'] / window, np.ones(window))

data3 = pd.read_csv('./result_alpha_0.5.csv')
reward3 = convolve1d(data3['steps'] / window, np.ones(window))

data4 = pd.read_csv('./result_alpha_0.7.csv')
reward4 = convolve1d(data4['steps'] / window, np.ones(window))


data5 = pd.read_csv('./result_alpha_0.9.csv')
reward5 = convolve1d(data5['steps'] / window, np.ones(window))

plt.plot(reward1, label='alpha=0.1')
plt.plot(reward2, label='alpha=0.3')
plt.plot(reward3, label='alpha=0.5')
plt.plot(reward4, label='alpha=0.7')
plt.plot(reward5, label='alpha=0.9')
plt.legend()
plt.xlabel('Number of Episodes')
plt.ylabel("Number of Steps")
# plt.show()
plt.tight_layout()
plt.savefig('./alpha_epi.png', dpi=300, quality=95)
# steps = np.arange(200)
# print(0.05 + (0.9 - 0.05) * np.exp(-1. * steps / 50))