from datanetAPI import DatanetAPI
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def get_data(path, max_num_samples):
    tool = DatanetAPI(path, shuffle=True)
    it = iter(tool)
    num_samples = 0
    traffic = []
    capacity = []
    path_len = []
    queue_occupancy = []
    for sample in it:
        G = nx.DiGraph(sample.get_topology_object())
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        P = sample.get_port_stats()
        for src in range(G.number_of_nodes()):
            for dst in range(G.number_of_nodes()):
                if src != dst:
                    traffic.append(T[src, dst]['AggInfo']['AvgBw'])
                    path_len.append(len(R[src, dst]) - 1)
                    if G.has_edge(src, dst):
                        capacity.append(G.edges[src, dst]['bandwidth'])
                        queue_occupancy.append((P[src][dst]['qosQueuesStats'][0]['avgPortOccupancy'] - 1) /
                                               G.nodes[src]['queueSizes'])
        print(num_samples)
        num_samples += 1
        if num_samples == max_num_samples:
            break
    return traffic, capacity, path_len, queue_occupancy


train_traffic, train_capacity, train_path_len, queue_occupancy_train = get_data('./data/train', 1000)
validation_traffic, validation_capacity, validation_path_len, queue_occupancy_test = get_data('./data/validation', 5)

plt.hist(np.log(queue_occupancy_train), bins=2000, alpha=0.5, label='Train')
plt.hist(np.log(queue_occupancy_test), bins=2000, alpha=0.5, label='Validation')
plt.legend()
plt.title('Occupancy')
plt.show()
plt.close()


plt.hist(train_traffic, bins=2000, alpha=0.5, label='Train')
plt.hist(validation_traffic, bins=2000, alpha=0.5, label='Validation')
plt.ylim(0, 8000)
plt.legend()
plt.title('Traffic')
plt.show()
plt.close()

plt.hist(train_capacity, bins=5, alpha=0.5, label='Train')
plt.hist(validation_capacity, bins=50, alpha=0.5, label='Validation')
plt.legend()
plt.title('Capacity')
plt.show()
plt.close()

plt.hist(train_path_len, bins=5, alpha=0.5, label='Train')
plt.hist(validation_path_len, bins=14, alpha=0.5, label='Validation')
plt.legend()
plt.title('Path length')
plt.show()
plt.close()
