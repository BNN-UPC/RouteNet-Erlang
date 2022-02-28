import tensorflow as tf


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
        readout (Keras Model): Readout Neural Network. It expects as input the
                               path states and outputs the per-path delay.
    """

    def __init__(self, config, output_units=1):
        super(GNN_Model, self).__init__()

        # Configuration dictionary. It contains the needed Hyperparameters for the model.
        # All the Hyperparameters can be found in the config.ini file
        self.config = config

        self.threshold = float(self.config['HYPERPARAMETERS']['threshold'])
        self.max_iter = int(self.config['HYPERPARAMETERS']['max_iter'])

        # GRU Cells used in the Message Passing step
        self.link_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['link_state_dim']))
        self.path_update = tf.keras.layers.GRUCell(int(self.config['HYPERPARAMETERS']['path_state_dim']))

        self.masking = tf.keras.layers.Masking()

        self.link_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=2),
            tf.keras.layers.Dense(int(int(self.config['HYPERPARAMETERS']['link_state_dim']) / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['link_state_dim']),
                                  activation=tf.keras.activations.relu)
        ])

        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=2),
            tf.keras.layers.Dense(int(int(self.config['HYPERPARAMETERS']['path_state_dim']) / 2),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['path_state_dim']),
                                  activation=tf.keras.activations.relu)
        ])

        self.aggr_mlp = tf.keras.Sequential([
            tf.keras.layers.Input(shape=4 * int(self.config['HYPERPARAMETERS']['path_state_dim'])),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['path_state_dim']),
                                  activation=tf.keras.activations.relu)
        ])

        # Readout Neural Network. It expects as input the path states and outputs the per-path delay
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Input(shape=int(self.config['HYPERPARAMETERS']['link_state_dim'])),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.keras.activations.relu,
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
            tf.keras.layers.Dense(int(self.config['HYPERPARAMETERS']['readout_units']),
                                  activation=tf.keras.activations.relu,
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
            tf.keras.layers.Dense(output_units, activation=tf.keras.activations.sigmoid)
        ])

    @tf.function
    def call(self, inputs):
        """This function is execution each time the model is called
        Args:
            inputs (dict): Features used to make the predictions.
            training (bool): Whether the model is training or not. If False, the
                             model does not update the weights.
        Returns:
            tensor: A tensor containing the per-path delay.
        """

        traffic = tf.expand_dims(tf.squeeze(inputs['traffic']), axis=1)
        packets = tf.expand_dims(tf.squeeze(inputs['packets']), axis=1)
        capacity = tf.expand_dims(tf.squeeze(inputs['capacity']), axis=1)
        scale = tf.expand_dims(tf.squeeze(inputs['scale']), axis=1)
        link_to_path = tf.squeeze(inputs['link_to_path'])
        path_ids = tf.squeeze(inputs['path_ids'])
        sequence_path = tf.squeeze(inputs['sequence_path'])
        sequence_links = tf.squeeze(inputs['sequence_links'])
        n_links = inputs['n_links']
        n_paths = inputs['n_paths']

        # Initialize the initial hidden state for links
        link_shape = tf.stack([
            n_links,
            int(self.config['HYPERPARAMETERS']['link_state_dim'])
        ], axis=0)

        link_state = tf.concat([
            capacity,
            scale
        ], axis=1)

        link_state = self.link_embedding(link_state)

        # Initialize the initial hidden state for paths
        path_state = tf.concat([
            traffic,
            packets
        ], axis=1)

        initial_path_state = self.path_embedding(path_state)

        for _ in range(8):
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
                                                              initial_state=initial_path_state)

            # For every link, gather and compute the aggr. of the sequence of hidden states of the paths that contain it
            path_gather = tf.gather_nd(path_state_sequence, ids)
            path_sum = tf.math.unsorted_segment_sum(path_gather, sequence_links, n_links)
            path_mean = tf.math.unsorted_segment_mean(path_gather, sequence_links, n_links)
            path_max = tf.math.unsorted_segment_max(path_gather, sequence_links, n_links)
            path_max = tf.where(tf.equal(tf.float32.min, path_max), tf.zeros_like(path_max), path_max)
            path_min = tf.math.unsorted_segment_min(path_gather, sequence_links, n_links)
            path_min = tf.where(tf.equal(tf.float32.max, path_min), tf.zeros_like(path_min), path_min)

            mlp_input = tf.concat([
                path_sum,
                path_mean,
                path_max,
                path_min
            ], axis=1)

            mlp_input = tf.ensure_shape(mlp_input,
                                        [None, 4 * int(self.config['HYPERPARAMETERS']['path_state_dim'])])
            path_aggregation = self.aggr_mlp(mlp_input)
            link_state, _ = self.link_update(path_aggregation, [link_state])

        ids = tf.stack([path_ids, sequence_path], axis=1)
        max_len = tf.reduce_max(sequence_path) + 1
        shape = tf.stack([
            n_paths,
            max_len,
            1])

        # Call the readout ANN
        occupancy = self.readout(link_state)

        occupancy_gather = tf.gather(occupancy, link_to_path)
        occupancy = tf.scatter_nd(ids, occupancy_gather, shape)

        # Denormalize bandwidth and scale features
        bandwidth_mean = 21166.35
        bandwidth_std = 24631.01
        scale_mean = 10.5
        scale_std = 5.77

        real_scale = scale * scale_std + scale_mean
        real_cap = capacity * bandwidth_std + bandwidth_mean
        real_cap = real_cap * real_scale

        capacity_gather = tf.gather(real_cap, link_to_path)
        capacity = tf.scatter_nd(ids, capacity_gather, shape)
        # capacity = tf.where(tf.equal(capacity, 0), tf.ones_like(capacity), capacity)

        # Compute the delay given the queue occupancy and link capacities
        queueing_delay = (occupancy * 32 * 1000) / capacity
        queueing_delay = tf.where(tf.math.is_nan(queueing_delay), tf.zeros_like(queueing_delay), queueing_delay)
        queueing_delay = tf.math.reduce_sum(queueing_delay, axis=1)

        trans_delay = 1000 / capacity
        trans_delay = tf.where(tf.math.is_inf(trans_delay), tf.zeros_like(trans_delay), trans_delay)
        trans_delay = tf.math.reduce_sum(trans_delay, axis=1)
        return queueing_delay + trans_delay
