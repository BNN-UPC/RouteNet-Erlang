import sys

sys.path.insert(1, './TrafficModel')
sys.path.insert(1, './')
import numpy as np
from read_dataset import input_fn
import matplotlib.pyplot as plt

ds = input_fn('./data/time-dist-experiments', label='jitter', shuffle=True)
for elem in ds.take(5000):
    print(np.log(elem[1]))

import sys
sys.path.insert(1, './data')
from datanetAPI_all import DatanetAPI
import networkx as nx
import pickle
EXTERNAL_DISTRIBUTIONS = ['AR1-0', 'AR1-1']

api = DatanetAPI('./data/Mixed', shuffle=True)
it = iter(api)
sample = next(it)
data_dict = {}
num_samples = 0
for sample in it:
    G = nx.DiGraph(sample.get_topology_object())
    R = sample.get_routing_matrix()
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                flow = sample.get_traffic_matrix()[src, dst]['Flows'][0]
                if flow['TimeDist'].value != 6:
                    name = flow['TimeDist'].name
                else:
                    name = flow['TimeDistParams']['Distribution']
                if name not in data_dict:
                    data_dict[name] = {}

                for param in flow['TimeDistParams']:
                    if param != 'Distribution':
                        if param not in data_dict[name]:
                            data_dict[name][param] = []
                        data_dict[name][param].append(flow['TimeDistParams'][param])
    num_samples += 1
    if num_samples%100==0:
        print("Saving for iterations: {}".format(num_samples))
        a_file = open("data.pkl", "wb")
        pickle.dump(data_dict, a_file)
        a_file.close()

a_file = open("data.pkl", "rb")
output = pickle.load(a_file)
print(output)

s=[]
for a in output['AR1-0']['AR-a']:
    s.append((1.5/np.sqrt(1-a**2))**2)