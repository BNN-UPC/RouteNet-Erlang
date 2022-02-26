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
import random
import math
import networkx as nx
from datanetAPI import DatanetAPI


def generator(data_dir, min_scale, max_scale, samples_per_sample=1, shuffle=False):
    try:
        data_dir = data_dir.decode('UTF-8')
    except (UnicodeDecodeError, AttributeError):
        pass
    tool = DatanetAPI(data_dir, shuffle=shuffle)
    it = iter(tool)
    num_samples = 0
    for sample in it:
        for _ in range(samples_per_sample):
            num_samples += 1
            # print(num_samples)
            G_copy = sample.get_topology_object().copy()
            T = sample.get_traffic_matrix()
            R = sample.get_routing_matrix()
            D = sample.get_performance_matrix()
            P = sample.get_port_stats()
            HG = network_to_hypergraph(network_graph=G_copy,
                                       routing_matrix=R,
                                       traffic_matrix=T,
                                       performance_matrix=D,
                                       port_stats=P,
                                       min_scale=min_scale,
                                       max_scale=max_scale)

            ret = hypergraph_to_input_data(HG)
            if not all(d >= 0 for d in ret[1]):
                continue

            yield ret


def network_to_hypergraph(network_graph, routing_matrix, traffic_matrix, performance_matrix, port_stats,
                          min_scale, max_scale):
    G = nx.DiGraph(network_graph)
    R = routing_matrix
    T = traffic_matrix
    D = performance_matrix
    P = port_stats

    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                if G.has_edge(src, dst):
                    scale = np.random.randint(min_scale, max_scale)
                    D_G.add_node('l_{}_{}'.format(src, dst),
                                 capacity=G.edges[src, dst]['bandwidth'] / scale,
                                 scale=scale,
                                 occupancy=(P[src][dst]['qosQueuesStats'][0]['avgPortOccupancy'] - 1) /
                                           G.nodes[src]['queueSizes'],
                                 queue_size=int(G.nodes[src]['queueSizes']))

                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                        D_G.add_node('p_{}_{}_{}'.format(src, dst, f_id),
                                     traffic=T[src, dst]['Flows'][f_id]['AvgBw'],
                                     packets=T[src, dst]['Flows'][f_id]['PktsGen'],
                                     delay=D[src, dst]['Flows'][f_id]['AvgDelay'])

                        for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                            D_G.add_edge('p_{}_{}_{}'.format(src, dst, f_id), 'l_{}_{}'.format(h_1, h_2))
                            D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'p_{}_{}_{}'.format(src, dst, f_id))

    D_G.remove_nodes_from([node for node, out_degree in D_G.out_degree() if out_degree == 0])

    return D_G


def hypergraph_to_input_data(hypergraph):
    n_p = 0
    n_l = 0
    mapping = {}
    for entity in list(hypergraph.nodes()):
        if entity.startswith('p'):
            mapping[entity] = ('p_{}'.format(n_p))
            n_p += 1
        elif entity.startswith('l'):
            mapping[entity] = ('l_{}'.format(n_l))
            n_l += 1

    D_G = nx.relabel_nodes(hypergraph, mapping)

    link_to_path = []
    path_ids = []
    sequence_path = []
    out_degree = []
    for i in range(n_p):
        seq_len = 0
        for elem in D_G['p_{}'.format(i)]:
            link_to_path.append(int(elem.replace('l_', '')))
            seq_len += 1
        path_ids.extend(np.full(seq_len, i))
        sequence_path.extend(range(seq_len))
        out_degree.append(D_G.out_degree('p_{}'.format(i)))

    path_to_link = []
    sequence_links = []
    for i in range(n_l):
        seq_len = 0
        for elem in D_G['l_{}'.format(i)]:
            path_to_link.append(int(elem.replace('p_', '')))
            seq_len += 1
        sequence_links.extend(np.full(seq_len, i))

    return {"traffic": list(nx.get_node_attributes(D_G, 'traffic').values()),
            "packets": list(nx.get_node_attributes(D_G, 'packets').values()),
            "out_degree": out_degree,
            "occupancy": list(nx.get_node_attributes(D_G, 'occupancy').values()),
            "capacity": list(nx.get_node_attributes(D_G, 'capacity').values()),
            "queue_size": list(nx.get_node_attributes(D_G, 'queue_size').values()),
            "scale": list(nx.get_node_attributes(D_G, 'scale').values()),
            "link_to_path": link_to_path,
            "path_to_link": path_to_link,
            "path_ids": path_ids,
            "sequence_links": sequence_links,
            "sequence_path": sequence_path,
            "n_links": n_l,
            "n_paths": n_p
            }, list(nx.get_node_attributes(D_G, 'delay').values())


def input_fn(data_dir, min_scale, max_scale, samples_per_sample=1, shuffle=False, samples=None):
    ds = tf.data.Dataset.from_generator(generator,
                                        args=[data_dir, min_scale, max_scale, samples_per_sample, shuffle],
                                        output_types=(
                                            {"traffic": tf.float32,
                                             "packets": tf.float32,
                                             "out_degree": tf.int32,
                                             "capacity": tf.float32,
                                             "queue_size": tf.int32,
                                             "scale": tf.float32,
                                             "occupancy": tf.float32,
                                             "link_to_path": tf.int32,
                                             "path_to_link": tf.int32, "path_ids": tf.int32,
                                             "sequence_links": tf.int32, "sequence_path": tf.int32,
                                             "n_links": tf.int32, "n_paths": tf.int32},
                                            tf.float32),
                                        output_shapes=(
                                            {"traffic": tf.TensorShape([None]),
                                             "packets": tf.TensorShape([None]),
                                             "out_degree": tf.TensorShape([None]),
                                             "capacity": tf.TensorShape([None]),
                                             "occupancy": tf.TensorShape([None]),
                                             "queue_size": tf.TensorShape([None]),
                                             "scale": tf.TensorShape([None]),
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
