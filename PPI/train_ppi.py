# %%

from time import time
from pathlib import Path

import tensorflow as tf
from absl import flags
import scipy.sparse as sp
import numpy as np
from tqdm.autonotebook import tqdm

from PPI.utils import load_edgelist_data, sparse_to_info, process_graph, split_graph_edges
from model.gcn import GCN

# %%

# Set hyper-parameters.
FLAGS = flags.FLAGS
flags.DEFINE_float(name='lr', default=0.01, help='learning rate', lower_bound=0.)
flags.DEFINE_float(name='val_r', default=0.02, help='Validation set ratio', lower_bound=0., upper_bound=1.)
flags.DEFINE_float(name='test_r', default=0.02, help='Test set ratio', lower_bound=0., upper_bound=1.)
flags.DEFINE_integer(name='seed', default=None, help='Seed for data split. Not the seed for model training.')
flags.DEFINE_float(name='dropout', default=0.5, help='Dropout rate.', lower_bound=0, upper_bound=1)
flags.DEFINE_bool(name='bias', default=True, help='Whether to use bias in GCN model.')
flags.DEFINE_integer(name='h1', default=16, help='Number of hidden features for the first hidden layer of the GCN,'
                                                 ' which has only 2 hidden layers.', lower_bound=1)
flags.DEFINE_integer(name='h2', default=16, help='Number of hidden features for the second hidden layer of the GCN,'
                                                 ' which has only 2 hidden layers.', lower_bound=1)
flags.DEFINE_integer(name='epochs', default=100, help='Number of training epochs.', lower_bound=1)
flags.DEFINE_string(name='logs', default='./logs',
                    help='root directory for logs, checkpoints, and other records.')
FLAGS.mark_as_parsed()  # This is necessary for using FLAGS in jupyter.

# %%

# TODO: Please note that sparse dropout is not implemented properly for graph mode.

data_path = '../data/yeast.edgelist'
adj = load_edgelist_data(data_path)  # Get adjacency matrix in CSR format.

# The adjacency matrix is symmetric but has non-zero elements on the diagonal.
num_nodes = adj.shape[0]
num_edges = adj.sum()


# %%

adj_orig = adj - sp.diags(adj.diagonal())  # Removing diagonal elements.
adj_orig.eliminate_zeros()

# %%

data = split_graph_edges(adj, val_ratio=FLAGS.val_r, test_ratio=FLAGS.test_r, seed=FLAGS.seed)
adj_train, edges_train, edges_val, edges_val_fake, edges_test, edges_test_fake = data

norm_edges, norm_data, norm_shape = process_graph(adj_train)
adj_train_norm = tf.SparseTensor(indices=norm_edges, values=norm_data, dense_shape=norm_shape)


# %%

model = GCN(num_features=num_nodes, h1=FLAGS.h1, h2=FLAGS.h2, dropout=FLAGS.dropout, bias=FLAGS.bias)

optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
log_dir = Path(FLAGS.logs)
log_dir.mkdir(exist_ok=True)
log_dir /= f'log_{len(list(log_dir.iterdir())) + 1}'
writer = tf.summary.create_file_writer(logdir=str(log_dir))

# %%

lab_edges, lab_data, lab_shape = sparse_to_info(adj_train + sp.identity(num_nodes, dtype=np.float32))
adj_labels = tf.sparse.to_dense(tf.SparseTensor(indices=lab_edges, values=lab_data, dense_shape=lab_shape))

oh_edges, oh_data, oh_shape = sparse_to_info(sp.identity(num_nodes, dtype=np.float32))
one_hot = tf.SparseTensor(indices=oh_edges, values=oh_data, dense_shape=oh_shape)

position_weight = (num_nodes ** 2 - num_edges) / num_edges
norm = num_nodes ** 2 / (2 * (num_nodes ** 2 - num_edges))

# %%

with writer.as_default():
    # Training loop
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):  # Only 1 training iteration per epoch.
        tic = time()
        inputs = [one_hot, adj_train_norm]

        with tf.GradientTape() as tape:  # Begin gradient calculations from here.
            outputs = model(inputs, training=True)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=adj_labels, logits=outputs))
        variables = model.trainable_variables
        gradients = tape.gradient(target=loss, sources=variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, variables))

        toc = int(time() - tic)
        # print(f'Epoch {epoch:03d} Time: {toc}s, loss: {float(loss):.3f}')
        tf.summary.scalar(name='Training Loss', data=loss, step=epoch, description='Training loss for each epoch.')
    else:
        writer.flush()

