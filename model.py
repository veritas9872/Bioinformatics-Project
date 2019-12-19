import tensorflow as tf
from tensorflow.keras.layers import Layer, InputLayer
from tensorflow.keras import Model, activations

def dropout_sparse(x, prob, num_nonzero):
    random_tensor = tf.random.uniform(shape=[num_nonzero]) + 1.-prob
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    out = tf.sparse.retain(x, dropout_mask)
    
    return out / prob
   
class GraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, dropout=0., activation='linear'):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activations.get(activation)
        
    def build(self, input_shape):
        self.w = self.add_weight(name='weight',
                                 shape=(self.input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
    def call(self, inputs, training = None):
        tensor, adj = inputs
        x = tensor
        if training:
            x = tf.nn.dropout(x, self.dropout)
        x = tf.matmul(x, self.w)
        x = tf.sparse.sparse_dense_matmul(adj, x)
        
        return self.activation(x)

class GraphConvolutionSparse(Layer):
    def __init__(self, input_dim, output_dim, num_nonzeros, dropout=0., activation='relu'):
        super(GraphConvolutionSparse, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nonzeros = num_nonzeros
        self.dropout = dropout
        self.activation = activations.get(activation)
        
    def build(self, input_shape):
        self.w = self.add_weight(name='weight',
                                 shape=(self.input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
    def call(self, inputs, training = None):
        tensor, adj = inputs
        x = tensor
        if training:
            x = dropout_sparse(x, self.dropout, self.num_nonzeros)
        x = tf.sparse.sparse_dense_matmul(x, self.w)
        x = tf.sparse.sparse_dense_matmul(adj, x)
        return self.activation(x)
    
class InnerProductDecoder(Layer):
    def __init__(self, dropout=0.):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        
    def __call__(self, inputs):
        x = tf.nn.dropout(inputs, self.dropout)
        x_tp = tf.transpose(x)
        x = tf.matmul(x, x_tp)
        
        return x        

class GCN(Model):
    def __init__(self, num_features, num_nonzeros, num_h1, num_h2, dropout=0.):
        super(GCN, self).__init__(name='gcn')
        self.num_features = num_features
        self.num_nonzeros = num_nonzeros
        self.dropout = dropout
        self.num_h1 = num_h1
        self.num_h2 = num_h2    
        
        self.hidden1 = GraphConvolutionSparse(self.num_features,
                                              self.num_h1,
                                              self.num_nonzeros,
                                              dropout=self.dropout)
        self.hidden2 = GraphConvolution(self.num_h1,
                                        self.num_h2,
                                        dropout=self.dropout)
        self.recon = InnerProductDecoder(dropout=self.dropout)
        
    def call(self, inputs, training = None):
        _ , adj= inputs            
        x = self.hidden1(inputs, training=training)
        x = self.hidden2([x, adj], training=training)     
        x = self.recon(x)
        
        return tf.cast(x, tf.float64)
    