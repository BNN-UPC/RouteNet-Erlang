from datanetAPI import DatanetAPI
import configparser
from glob import iglob
import os

config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')

# Ensure we read the files in the correct order
directories = [d for d in iglob(config['DIRECTORIES']['test'] + '/*/*')]
# First, sort by scenario and second, by topology size
directories.sort(key=lambda f: (os.path.dirname(f), int(os.path.basename(f))))
ground_truth_file = open("ground_truth_sample_data.txt", "w")
path_per_sample = open("path_per_sample_sample_data.txt", "w")

num_samples = 0
first = True
for d in directories:
    api = DatanetAPI(d, shuffle=False)
    it = iter(api)
    for sample in it:
        if not first:
            ground_truth_file.write("\n")
            path_per_sample.write("\n")
        G = sample.get_topology_object()
        D = sample.get_performance_matrix()
        T = sample.get_traffic_matrix()
        delay = []
        for src in range(G.number_of_nodes()):
            for dst in range(G.number_of_nodes()):
                if src != dst:
                    for f_id in range(len(T[src, dst]['Flows'])):
                        if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                            delay.append(D[src, dst]['Flows'][f_id]['AvgDelay'])

        ground_truth_file.write("{}".format(';'.join([str(i) for i in delay])))
        path_per_sample.write("{}".format(len(delay)))
        first = False
        num_samples += 1
        print(num_samples)

ground_truth_file.close()
path_per_sample.close()

first=True
with open('test.txt','r') as test, open('test_truncated.txt','w') as test_trunc:
    for elem in test:
        if not first:
            test_trunc.write("\n")
        elem = elem.rstrip()

        # SPLIT THE LIST, CONVERT TO FLOAT AND THEN TO LIST
        elem = list(map(float, elem.split(";")))
        test_trunc.write("{}".format(';'.join([format(i,'.6f') for i in elem])))
        first = False

