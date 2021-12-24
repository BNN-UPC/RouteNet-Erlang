import sys
sys.path.insert(1, './TrafficModel')
from datanetAPI import DatanetAPI
import pandas as pd
import traceback

api = DatanetAPI('./data/time-dist-experiments/k2/test')
it = iter(api)
delay = []
jitter = []
traffic = []
drops = []
num_samples = 0
for sample in it:
    try:
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
                    aux_traffic.append(T[src, dst]['Flows'][0]['AvgBw'])
                    aux_delay.append(P[src, dst]['AggInfo']['AvgDelay'])
                    aux_jitter.append(P[src, dst]['AggInfo']['Jitter'])
                    aux_drops.append(float(P[src, dst]['AggInfo']['PktsDrop']) / float(T[src, dst]['Flows'][0]['PktsGen']))

        if 0 in aux_delay or 0 in aux_jitter:
            continue
        delay.extend(aux_delay)
        traffic.extend(aux_traffic)
        jitter.extend(aux_jitter)
        drops.extend(aux_drops)
        num_samples += 1
        print(num_samples)
    except Exception as e:
        print(traceback.format_exc())

df = pd.DataFrame(
    {"traffic": traffic, "delay": delay, "jitter": jitter, "drops": drops})

df.to_feather("k2_dataframe")
