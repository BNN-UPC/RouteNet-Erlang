"""
   Copyright 2020 Universitat Politècnica de Catalunya
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import tensorflow as tf
import networkx as nx
from datanetAPI import DatanetAPI

EXTERNAL_DISTRIBUTIONS = ['AR1-0', 'AR1-1']


def generator(data_dir, label, shuffle=False):
    tool = DatanetAPI(data_dir, shuffle=shuffle)
    it = iter(tool)
    num_samples = 0
    for sample in it:
        try:
            HG = network_to_hypergraph(sample=sample)
            num_samples += 1
            n_p = 0
            n_l = 0
            mapping = {}
            for entity in list(HG.nodes()):
                if entity.startswith('p'):
                    mapping[entity] = ('p_{}'.format(n_p))
                    n_p += 1
                elif entity.startswith('l'):
                    mapping[entity] = ('l_{}'.format(n_l))
                    n_l += 1

            D_G = nx.relabel_nodes(HG, mapping)

            link_to_path = []
            path_ids = []
            sequence_path = []
            for i in range(n_p):
                seq_len = 0
                for elem in D_G['p_{}'.format(i)]:
                    link_to_path.append(int(elem.replace('l_', '')))
                    seq_len += 1
                path_ids.extend(np.full(seq_len, i))
                sequence_path.extend(range(seq_len))

            path_to_link = []
            sequence_links = []
            for i in range(n_l):
                seq_len = 0
                for elem in D_G['l_{}'.format(i)]:
                    path_to_link.append(int(elem.replace('p_', '')))
                    seq_len += 1
                sequence_links.extend(np.full(seq_len, i))

            if 0 in list(nx.get_node_attributes(D_G, 'jitter').values()) or 0 in list(
                    nx.get_node_attributes(D_G, 'delay').values()):
                continue

            yield {"traffic": list(nx.get_node_attributes(D_G, 'traffic').values()),
                   "packets": list(nx.get_node_attributes(D_G, 'packets').values()),
                   "time_dist_params": list(nx.get_node_attributes(D_G, 'time_dist_params').values()),
                   "capacity": list(nx.get_node_attributes(D_G, 'capacity').values()),
                   "link_to_path": link_to_path,
                   "path_to_link": path_to_link,
                   "path_ids": path_ids,
                   "sequence_links": sequence_links,
                   "sequence_path": sequence_path,
                   "n_links": n_l,
                   "n_paths": n_p
                   }, list(nx.get_node_attributes(D_G, label).values())

        except Exception as e:
            pass

def network_to_hypergraph(sample):
    G = nx.DiGraph(sample.get_topology_object())
    R = sample.get_routing_matrix()
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()

    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:

                time_dist_params = [0] * 12
                flow = sample.get_traffic_matrix()[src, dst]['Flows'][0]
                if flow['TimeDist'].value != 6:
                    time_dist_params[flow['TimeDist'].value] = 1
                else:
                    time_dist_params[flow['TimeDist'].value + EXTERNAL_DISTRIBUTIONS.index(
                        flow['TimeDistParams']['Distribution'])] = 1

                idx = 7
                for k in flow['TimeDistParams']:
                    if isinstance(flow['TimeDistParams'][k], int) or isinstance(flow['TimeDistParams'][k], float):
                        time_dist_params[idx] = flow['TimeDistParams'][k]
                        idx += 1

                D_G.add_node('p_{}_{}'.format(src, dst),
                             traffic=T[src, dst]['Flows'][0]['AvgBw'],
                             packets=T[src, dst]['Flows'][0]['PktsGen'],
                             source=src,
                             destination=dst,
                             time_dist_params=time_dist_params,
                             drops=float(P[src, dst]['AggInfo']['PktsDrop']) / float(
                                 T[src, dst]['Flows'][0]['PktsGen']),
                             delay=float(P[src, dst]['AggInfo']['AvgDelay']),
                             jitter=float(P[src, dst]['AggInfo']['Jitter']))

                if G.has_edge(src, dst):
                    D_G.add_node('l_{}_{}'.format(src, dst),
                                 capacity=G.edges[src, dst]['bandwidth'])

                for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                    D_G.add_edge('p_{}_{}'.format(src, dst), 'l_{}_{}'.format(h_1, h_2))
                    D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'p_{}_{}'.format(src, dst))

    D_G.remove_nodes_from([node for node, out_degree in D_G.out_degree() if out_degree == 0])

    return D_G

def input_fn(data_dir, label, shuffle=False, samples=None):
    ds = tf.data.Dataset.from_generator(lambda: generator(data_dir=data_dir, label=label, shuffle=shuffle),
                                        output_types=(
                                            {"traffic": tf.float32,
                                             "packets": tf.float32,
                                             "time_dist_params": tf.float32,
                                             "capacity": tf.float32,
                                             "link_to_path": tf.int32,
                                             "path_to_link": tf.int32, "path_ids": tf.int32,
                                             "sequence_links": tf.int32, "sequence_path": tf.int32,
                                             "n_links": tf.int32, "n_paths": tf.int32},
                                            tf.float32),
                                        output_shapes=(
                                            {"traffic": tf.TensorShape([None]),
                                             "packets": tf.TensorShape([None]),
                                             "time_dist_params": tf.TensorShape([None, None]),
                                             "capacity": tf.TensorShape([None]),
                                             "link_to_path": tf.TensorShape([None]),
                                             "path_to_link": tf.TensorShape([None]),
                                             "path_ids": tf.TensorShape([None]),
                                             "sequence_links": tf.TensorShape([None]),
                                             "sequence_path": tf.TensorShape([None]),
                                             "n_links": tf.TensorShape([]),
                                             "n_paths": tf.TensorShape([])},
                                            tf.TensorShape([None])))

    if samples:
        ds = ds.take(samples)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds
