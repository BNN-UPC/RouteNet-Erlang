import sys

sys.path.insert(1, './data/')
import numpy as np
from read_dataset import input_fn
import matplotlib.pyplot as plt

ds = input_fn('./data/time-dist-experiments', label='jitter', shuffle=True)
for elem in ds.take(5):
    print(elem[1])

jitter = []
num_samples = 0
for x, y in ds.take(200):
    jitter.extend(y)
    num_samples += 1
    print(num_samples)

plt.hist(jitter,bins=500)
plt.xlim(-0.05, 0.05)
plt.show()
plt.close()

plt.hist(jitter, bins=5000)
plt.xlim(-1, 1.5)
plt.show()
plt.close()

plt.hist(np.log(jitter), bins=5000)
plt.show()
plt.close()
