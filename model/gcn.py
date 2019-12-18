import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, Model


# This function causes errors when used in graph mode.
def sparse_dropout(sparse_tensor: tf.sparse.SparseTensor, rate: float, num_features: int) -> tf.sparse.SparseTensor:
    if rate == 0:
        return sparse_tensor
    assert 0 < rate < 1, 'Invalid range for dropout rate.'
    random_tensor = tf.random.uniform(shape=sparse_tensor.values.shape) + 1 - rate
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
    def __init__(self, input_dim: int, output_dim: int, bias=True,
                 dropout=0., is_sparse=False, activation=None, **kwargs):

        super().__init__(trainable=True, name='GraphConvolution', **kwargs)
        assert 0 <= dropout < 1, 'Invalid range for dropout.'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.dropout = dropout
        self.is_sparse = is_sparse
        self.activation = activations.get(activation)
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(name='weight', shape=(self.input_dim, self.output_dim),
                                 initializer='he_uniform', trainable=True)
        if self.bias:
            self.b = self.add_weight(name='bias', shape=(self.output_dim,), initializer='zeros', trainable=True)

        super().build(input_shape)

    def call(self, inputs: list, training: bool = None) -> tf.Tensor:
        assert len(inputs) == 2, 'Input must be a list of length 2 with the adjacency matrix as the second element.'
        tensor, adjacency_matrix = inputs

        input_is_sparse = isinstance(tensor, tf.sparse.SparseTensor)

        if self.is_sparse != input_is_sparse:
            raise RuntimeError('Input tensor does not match sparsity description!')

        if training:
            if self.is_sparse:
                # Not sure if num_features=input_dim is correct. However, our input data is always a square matrix.
                tensor = sparse_dropout(tensor, rate=self.dropout, num_features=self.input_dim)
            else:
                tensor = tf.nn.dropout(tensor, rate=self.dropout)

        # Sparse matrix multiplication can be specified in matmul.
        if self.is_sparse:
            tensor = tf.sparse.sparse_dense_matmul(sp_a=tensor, b=self.w)
        else:
            tensor = tf.matmul(a=tensor, b=self.w)
        tensor = tf.sparse.sparse_dense_matmul(sp_a=adjacency_matrix, b=tensor)
        if self.bias:
            tensor += self.b

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
    def __init__(self, num_features: int, h1: int, h2: int, dropout=0., bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = num_features
        self.h1 = h1
        self.h2 = h2
        self.dropout = dropout
        self.bias = bias
        self.hidden1 = None
        self.hidden2 = None

    def build(self, input_shape):
        self.hidden1 = GraphConvolution(input_dim=self.num_features, output_dim=self.h1, bias=self.bias,
                                        dropout=self.dropout, is_sparse=True, activation=tf.nn.relu)

        self.hidden2 = GraphConvolution(input_dim=self.h1, output_dim=self.h2, bias=self.bias,
                                        dropout=self.dropout, is_sparse=False, activation=tf.keras.activations.linear)

    def call(self, inputs: list, training: bool = None, mask=None) -> tf.Tensor:
        assert len(inputs) == 2, 'Input must be a list of length 2 with the adjacency matrix as the second element.'
        _, adjacency_matrix = inputs
        outputs = self.hidden1(inputs, training=training)
        outputs = self.hidden2([outputs, adjacency_matrix], training=training)
        # Recall from linear algebra that A x A^T is always a symmetric matrix.
        # The output shape of this matrix becomes (num_nodes, num_nodes).
        return tf.matmul(outputs, outputs, transpose_b=True)
