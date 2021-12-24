import sys

sys.path.insert(1, './TrafficModel/')
import numpy as np
from datanetAPI import DatanetAPI
import pandas as pd

qt_df = pd.read_csv('./original_dataframes/results_gbn-k1-5-9.csv')
qt_df['jitter']=jitter
qt_df['delay']=delay
qt_df['a']=alpha
EXTERNAL_DISTRIBUTIONS = ['AR1-0', 'AR1-1']
api = DatanetAPI('./data/gbn-k1', shuffle=False)
it = iter(api)
delay = []
jitter = []
alpha = []
num_sample = 0
for sample in it:
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()
    G = sample.get_topology_object()
    aux_delay = []
    aux_jitter = []
    aux_traffic = []
    aux_drops = []
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                delay.append(P[src, dst]['AggInfo']['AvgDelay'])
                jitter.append(P[src, dst]['AggInfo']['Jitter'])
                alpha.append(T[src, dst]['Flows'][0]['TimeDistParams']['AR-a'])
    num_sample += 1
    print(num_sample)
    """if num_sample == 1:
        break"""

for i in range(20):
    time_dist_params = [0] * 13
    flow = sample.get_traffic_matrix()[0, 1]['Flows'][0]
    if flow['TimeDist'].value != 6:
        time_dist_params[flow['TimeDist'].value] = 1
    else:
        time_dist_params[
            flow['TimeDist'].value + EXTERNAL_DISTRIBUTIONS.index(flow['TimeDistParams']['Distribution'])] = 1

    idx = 8
    for k in flow['TimeDistParams']:
        if isinstance(flow['TimeDistParams'][k], int) or isinstance(flow['TimeDistParams'][k], float):
            time_dist_params[idx] = flow['TimeDistParams'][k]
            idx += 1
    print('-' * 10)
    print(i)
    print(flow['TimeDist'])
    print(flow['TimeDistParams'])
    print(time_dist_params)

import sys

sys.path.insert(1, './TrafficModel')
from read_dataset import input_fn

ds = input_fn('./data/multiple_time_dis/', label='AvgDelay', shuffle=True)

for elem in ds:
    pass

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

import tensorflow as tf

a = [[0., 1, 0, 0, 0, 0], [1, 0, 0, 10, 0, 0]]
b = [[-0.8719275], [0.870776176]]

tf.concat([a, b], axis=1)
