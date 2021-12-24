import sys

sys.path.insert(1, './data/')
import numpy as np
from read_dataset import input_fn
import matplotlib.pyplot as plt

ds = input_fn('./data/nsfnet-wfk-drr-sp-4000', label='PktsDrop', shuffle=True)

jitter = []
num_samples = 0
traffic = []
packets = []
capacity = []
drops = []
size = []
n_samples = 0
for x, y in ds.take(50):
    traffic.extend(x["traffic"])
    packets.extend(x["packets"])
    capacity.extend(x["capacity"])
    drops.extend(y)
    n_samples += 1
    print(n_samples)

print("traffic_mean = {}".format(np.mean(traffic)))
print("traffic_std = {}".format(np.std(traffic)))
print("packets_mean = {}".format(np.mean(packets)))
print("packets_std = {}".format(np.std(packets)))
print("capacity_mean = {}".format(np.mean(capacity)))
print("capacity_std = {}".format(np.std(capacity)))
print("drops_mean = {}".format(np.mean(drops)))
print("drops_std = {}".format(np.std(drops)))

plt.hist(drops, bins=500)
plt.ylim(0, 500)
plt.show()
plt.close()
len(drops)-np.count_nonzero(drops)
import sys
sys.path.insert(1, './data/')

from datanetAPI import DatanetAPI

api = DatanetAPI("./data/nsfnet-wfk-drr-sp-4000")
it = iter(api)

sample = next(it)

sample.get_performance_matrix()
sample.get_traffic_matrix()