import tensorflow as tf


class GNN_Model(tf.keras.Model):

    def __init__(self, config, output_units=1):
        super(GNN_Model, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        # GRU Cells used in the Message Passing step
        self.link_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['link_state_dim']))
        self.path_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['path_state_dim']))

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
    def call(self, inputs):
        traffic = tf.expand_dims(tf.squeeze(inputs['traffic']), axis=1)
        packets = tf.expand_dims(tf.squeeze(inputs['packets']), axis=1)
        time_dist_params = tf.squeeze(inputs['time_dist_params'])
        capacity = tf.expand_dims(tf.squeeze(inputs['capacity']), axis=1)
        link_to_path = tf.squeeze(inputs['link_to_path'])
        path_to_link = tf.squeeze(inputs['path_to_link'])
        path_ids = tf.squeeze(inputs['path_ids'])
        sequence_path = tf.squeeze(inputs['sequence_path'])
        sequence_links = tf.squeeze(inputs['sequence_links'])
        n_links = inputs['n_links']
        n_paths = inputs['n_paths']

        # Compute the shape for the  all-zero tensor for path_state
        link_shape = tf.stack([
            n_links,
            int(self.config['HYPERPARAMETERS']['link_state_dim']) - 1
        ], axis=0)

        # Initialize the initial hidden state for paths
        link_state = tf.concat([
            capacity,
            tf.zeros(link_shape)
        ], axis=1)

        path_shape = tf.stack([
            n_paths,
            int(self.config['HYPERPARAMETERS']['link_state_dim']) -
            14
        ], axis=0)

        # Initialize the initial hidden state for links
        path_state = tf.concat([
            traffic,
            packets,
            time_dist_params,
            tf.zeros(path_shape)
        ], axis=1)

        for _ in range(int(self.config['HYPERPARAMETERS']['t'])):
            # The following lines generate a tensor of dimensions [n_paths, max_len_path, dimension_link] with all 0
            # but the link hidden states
            link_gather = tf.gather(link_state, link_to_path)

            ids = tf.stack([path_ids, sequence_path], axis=1)
            max_len = tf.reduce_max(sequence_path) + 1
            shape = tf.stack([
                n_paths,
                max_len,
                int(self.config['HYPERPARAMETERS']['link_state_dim'])])

            # Generate the aforementioned tensor [n_paths, max_len_path, dimension_link]
            link_inputs = tf.scatter_nd(ids, link_gather, shape)

            path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                                  return_sequences=True,
                                                  return_state=True)

            path_state_sequence, path_state = path_update_rnn(inputs=self.masking(link_inputs),
                                                              initial_state=path_state)

            # For every link, gather and sum the sequence of hidden states of the paths that contain it
            path_gather = tf.gather(path_state, path_to_link)
            # path_gather = tf.gather_nd(path_state_sequence, ids)
            path_sum = tf.math.unsorted_segment_sum(path_gather, sequence_links, n_links)

            # Second message passing: update the link_state
            # The ensure shape is needed for Graph_compatibility
            path_sum = tf.ensure_shape(path_sum, [None, int(self.config['HYPERPARAMETERS']['link_state_dim'])])
            link_state, _ = self.link_update(path_sum, [link_state])

        # Call the readout ANN and return its predictions
        r = self.readout(path_state)

        return r