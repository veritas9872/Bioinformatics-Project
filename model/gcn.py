import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, Model


def sparse_dropout(sparse_tensor: tf.sparse.SparseTensor, rate: float,
                   noise_shape=None, seed=None) -> tf.sparse.SparseTensor:
    if rate == 0:
        return sparse_tensor
    assert 0 <= rate < 1, 'Invalid range for dropout rate.'
    random_tensor = tf.random.uniform(shape=noise_shape, seed=seed) + 1 - rate
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    output = tf.sparse.retain(sparse_tensor, to_retain=dropout_mask)
    return output / (1. - rate)


class GraphConvolution(Layer):
    """
    Graph Convolution layer for both dense and sparse inputs.
    For use in undirected graphs without edge labels.
    Only weights, no biases used.
    Dropout included inside the layer.
    """
    def __init__(self, input_dim: int, output_dim: int, num_nonzero_features: int = None,
                 dropout=0., is_sparse=False, activation='relu', **kwargs):

        super().__init__(trainable=True, name='GraphConvolution', **kwargs)
        assert 0 <= dropout < 1
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nonzero_features = num_nonzero_features
        self.dropout = dropout
        self.is_sparse = is_sparse
        self.activation = activations.get(activation)
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight(name='weight', shape=(self.input_dim, self.output_dim),
                                 initializer='he_uniform', trainable=True)

        super().build(input_shape)

    # @tf.function
    def call(self, inputs: list, training: bool = None):
        assert len(inputs) == 2, 'Input must be a list of length 2 with the adjacency matrix as the second element.'
        tensor, adjacency_matrix = inputs

        input_is_sparse = isinstance(tensor, tf.sparse.SparseTensor)
        if self.is_sparse ^ input_is_sparse:  # ^ indicates xor in Python.
            raise RuntimeError('Input tensor does not match sparsity description!')

        if training:
            if self.is_sparse:
                tensor = sparse_dropout(tensor, rate=self.dropout, noise_shape=self.num_nonzero_features)
            else:
                tensor = tf.nn.dropout(tensor, rate=self.dropout)

        # Sparse matrix multiplication can be specified in matmul.
        tensor = tf.matmul(a=tensor, b=self.w, a_is_sparse=self.is_sparse)
        tensor = tf.matmul(a=adjacency_matrix, b=tensor, a_is_sparse=True)
        return self.activation(tensor)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'input_dim': self.input_dim, 'output_dim': self.output_dim,
                       'dropout': self.dropout, 'is_sparse': self.is_sparse})
        return config

    def compute_output_shape(self, input_shape: list) -> list:
        return [input_shape[0], self.output_dim]


class GCN(Model):
    """
    A basic 2 layer Graph Convolutional Network model.
    """
    def __init__(self, num_features: int, num_nonzero_features: int, h1: int, h2: int, dropout=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = num_features
        self.num_nonzero_features = num_nonzero_features
        self.dropout = dropout
        self.hidden1 = None
        self.hidden2 = None
        self.h1 = h1
        self.h2 = h2

    def build(self, input_shape):
        self.hidden1 = GraphConvolution(
            input_dim=self.num_features, output_dim=self.h1, num_nonzero_features=self.num_nonzero_features,
            dropout=self.dropout, is_sparse=True, activation=tf.nn.relu)

        self.hidden2 = GraphConvolution(input_dim=self.h1, output_dim=self.h2, dropout=self.dropout,
                                        is_sparse=False, activation=tf.keras.activations.linear)

    # @tf.function
    def call(self, inputs: list, training=None, mask=None):
        assert len(inputs) == 2, 'Input must be a list of length 2 with the adjacency matrix as the second element.'
        _, adjacency_matrix = inputs
        outputs = self.hidden1(inputs, training=training)
        outputs = self.hidden2([outputs, adjacency_matrix], training=training)
        return tf.matmul(tf.transpose(outputs), outputs)

