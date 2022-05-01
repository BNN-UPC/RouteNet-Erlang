"""
   Copyright 2021 Universitat Politècnica de Catalunya

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

from __future__ import print_function
import tensorflow as tf
import time


class GNN_Model(tf.keras.Model):
    """ Init method for the custom model.

    Args:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        output_units (int): Output units for the last readout's layer.

    Attributes:
        config (dict): Python dictionary containing the diferent configurations
                       and hyperparameters.
        link_update (GRUCell): Link GRU Cell used in the Message Passing step.
        path_update (GRUCell): Path GRU Cell used in the Message Passing step.
        queue_update (GRUCell): Queue GRU Cell used in the Message Passing step.
        readout (Keras Model): Readout Neural Network. It expects as input the
                               path states and outputs the per-path delay.
    """

    def __init__(self, config, output_units=1):
        super(GNN_Model, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # GRU Cells used in the Message Passing step
        self.path_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['path_state_dim']))
        self.link_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['link_state_dim']))
        self.queue_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['queue_state_dim']))

        self.masking = tf.keras.layers.Masking()
        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Input(shape=int(self.config['HYPERPARAMETERS']['path_state_dim'])),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.nn.relu),
            tf.keras.layers.Dense(output_units)
        ])

    @tf.function
    def call(self, inputs, training=False):
        """This function is execution each time the model is called

        Args:
            inputs (dict): Features used to make the predictions.
            training (bool): Whether the model is train or not. If False, the
                             model does not update the weights.

        Returns:
            tensor: A tensor containing the per-path delay.
        """
        traffic = tf.squeeze(inputs['traffic'])
        packets = tf.squeeze(inputs['packets'])
        capacity = tf.squeeze(inputs['capacity'])
        policy = tf.squeeze(inputs['policy'])
        priority = tf.squeeze(inputs['priority'])
        weight = tf.squeeze(inputs['weight'])
        path_ids = tf.squeeze(inputs['path_ids'])
        l_q_p = tf.squeeze(inputs['l_q_p'])
        l_p_s = tf.squeeze(inputs['l_p_s'])
        link_to_path = tf.squeeze(inputs['link_to_path'])
        queue_to_path = tf.squeeze(inputs['queue_to_path'])
        queue_to_link = tf.squeeze(inputs['queue_to_link'])
        sequence_links = tf.squeeze(inputs['sequence_links'])
        l_q_l = tf.squeeze(inputs['l_q_l'])
        n_paths = inputs['n_paths']
        n_links = inputs['n_links']
        n_queues = inputs['n_queues']


        # Compute the shape for the  all-zero tensor for link_state
        path_shape = tf.stack([
            n_paths,
            int(self.config['HYPERPARAMETERS']['link_state_dim']) -
            2
        ], axis=0)

        # Initialize the initial hidden state for links
        path_state = tf.concat([
            tf.expand_dims(traffic, axis=1),
            tf.expand_dims(packets, axis=1),
            tf.zeros(path_shape)
        ], axis=1)

        # Compute the shape for the  all-zero tensor for path_state
        link_shape = tf.stack([
            n_links,
            int(self.config['HYPERPARAMETERS']['path_state_dim']) -
            int(self.config['DATASET']['num_policies']) -
            1
        ], axis=0)

        # Initialize the initial hidden state for paths
        link_state = tf.concat([
            tf.expand_dims(capacity, axis=1),
            tf.one_hot(policy, int(self.config['DATASET']['num_policies'])),
            tf.zeros(link_shape)
        ], axis=1)

        # Compute the shape for the  all-zero tensor for path_state
        queue_shape = tf.stack([
            n_queues,
            int(self.config['HYPERPARAMETERS']['path_state_dim']) -
            int(self.config['DATASET']['max_num_queues']) -
            1
        ], axis=0)

        # Initialize the initial hidden state for paths
        queue_state = tf.concat([
            tf.one_hot(priority, int(self.config['DATASET']['max_num_queues'])),
            tf.expand_dims(weight, axis=1),
            tf.zeros(queue_shape)
        ], axis=1)

        # Iterate t times doing the message passing
        for it in range(int(self.config['HYPERPARAMETERS']['t'])):
            ###################
            #  LINK AND QUEUE #
            #     TO PATH     #
            ###################
            link_gather = tf.gather(link_state, link_to_path)
            queue_gather = tf.gather(queue_state, queue_to_path)

            ids_q = tf.stack([path_ids, l_q_p], axis=1)
            ids_l = tf.stack([path_ids, l_p_s], axis=1)

            max_len = tf.reduce_max(l_q_p) + 1
            shape = tf.stack([n_paths, max_len, int(self.config['HYPERPARAMETERS']['path_state_dim'])])

            queue_input = tf.scatter_nd(ids_q, queue_gather, shape)
            link_input = tf.scatter_nd(ids_l, link_gather, shape)

            path_gru_rnn = tf.keras.layers.RNN(self.path_update, return_sequences=False)

            path_state = path_gru_rnn(inputs=self.masking(tf.concat([queue_input, link_input], axis=2)),
                                      initial_state=path_state)

            ###################
            #  PATH TO QUEUE  #
            ###################
            path_gather = tf.gather(path_state, inputs['path_to_queue'])
            path_sum = tf.math.unsorted_segment_sum(path_gather, inputs['sequence_queues'], n_queues)
            queue_state, _ = self.queue_update(path_sum, [queue_state])

            ###################
            #  QUEUE TO LINK  #
            ###################
            queue_gather = tf.gather(queue_state, queue_to_link)
            ids_q = tf.stack([sequence_links, l_q_l], axis=1)
            max_len = tf.reduce_max(l_q_l) + 1
            shape = tf.stack([n_links, max_len, int(self.config['HYPERPARAMETERS']['link_state_dim'])])
            queue_input = tf.scatter_nd(ids_q, queue_gather, shape)

            link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
            link_state = link_gru_rnn(inputs=self.masking(queue_input), initial_state=link_state)

        # Call the readout ANN and return its predictions
        r = self.readout(path_state, training=training)

        return r
