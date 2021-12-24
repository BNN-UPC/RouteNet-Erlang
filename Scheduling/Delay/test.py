import sys
sys.path.insert(1, '../../data/')
import numpy as np
from read_dataset import input_fn

ds = input_fn('../../data/gnnet_data_set_training', label='AvgDelay', shuffle=True)

traffic = []
packets = []
capacity = []
size = []
n_samples = 0
for x, y in ds.take(50000):
    traffic.extend(x["traffic"])
    packets.extend(x["packets"])
    capacity.extend(x["capacity"])
    n_samples += 1
    print(n_samples)

print("traffic_mean = {}".format(np.mean(traffic)))
print("traffic_std = {}".format(np.std(traffic)))
print("packets_mean = {}".format(np.mean(packets)))
print("packets_std = {}".format(np.std(packets)))
print("capacity_mean = {}".format(np.mean(capacity)))
print("capacity_std = {}".format(np.std(capacity)))
