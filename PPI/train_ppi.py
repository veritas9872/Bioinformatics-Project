# %%

from time import time
from pathlib import Path

import tensorflow as tf
from absl import flags
import scipy.sparse as sp
import numpy as np

from PPI.utils import load_edgelist_data, to_normalized_sparse_tensor, \
    split_graph_edges, get_roc_score, to_sparse_tensor
from model.gcn import GCN

# Set hyper-parameters.
FLAGS = flags.FLAGS
flags.DEFINE_float(name='lr', default=0.01, help='learning rate', lower_bound=0.)
flags.DEFINE_float(name='val_r', default=0.02, help='Validation set ratio', lower_bound=0., upper_bound=1.)
flags.DEFINE_float(name='test_r', default=0.02, help='Test set ratio', lower_bound=0., upper_bound=1.)
flags.DEFINE_integer(name='seed', default=9872, help='Seed for data split. Not the seed for model training.')
flags.DEFINE_float(name='dropout', default=0.5, help='Dropout rate.', lower_bound=0, upper_bound=1)
flags.DEFINE_bool(name='bias', default=True, help='Whether to use bias in GCN model.')
flags.DEFINE_float(name='l2', default=0.005, help='L2 weight decay factor for weights and biases of the model.')
flags.DEFINE_integer(name='h1', default=4, lower_bound=1,
                     help='Number of hidden features for the first hidden layer of the GCN.')
flags.DEFINE_integer(name='h2', default=4, lower_bound=1,
                     help='Number of hidden features for the second hidden layer of the GCN.')
flags.DEFINE_integer(name='epochs', default=12, help='Number of training epochs.', lower_bound=1)
flags.DEFINE_string(name='logs', default='./logs',
                    help='root directory for logs, checkpoints, and other records.')
FLAGS.mark_as_parsed()  # This is necessary for using FLAGS in jupyter.

# %%

# NOTE: Sparse dropout is not implemented properly for graph mode.

data_path = '../data/yeast.edgelist'
adj = load_edgelist_data(data_path)  # Get adjacency matrix in CSR format.

# The adjacency matrix is symmetric but has non-zero elements on the diagonal.
num_nodes = adj.shape[0]
num_edges = adj.sum()

# %%

adj_orig = adj - sp.diags(adj.diagonal())  # Removing diagonal elements.
adj_orig.eliminate_zeros()

# %%

adj_train, edges_train, edges_val, edges_val_fake, edges_test, edges_test_fake = \
    split_graph_edges(adj, val_ratio=FLAGS.val_r, test_ratio=FLAGS.test_r, seed=FLAGS.seed)

adj_train_norm = to_normalized_sparse_tensor(adj_train)


# %%

model = GCN(num_features=num_nodes, h1=FLAGS.h1, h2=FLAGS.h2, dropout=FLAGS.dropout, bias=FLAGS.bias)

optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
log_dir = Path(FLAGS.logs)
log_dir.mkdir(exist_ok=True)
log_dir /= f'log_{len(list(log_dir.iterdir())) + 1}'
writer = tf.summary.create_file_writer(logdir=str(log_dir))

# %%

adj_train_labels = adj_train + sp.identity(num_nodes, dtype=np.float32)
adj_train_labels = tf.convert_to_tensor(adj_train_labels.todense())

one_hot = to_sparse_tensor(sp.identity(num_nodes, dtype=np.float32))

position_weight = (num_nodes ** 2 - num_edges) / num_edges
norm = num_nodes ** 2 / (2 * (num_nodes ** 2 - num_edges))

# %%

print('Beginning Training!')
with writer.as_default():
    # Training loop
    for epoch in range(1, FLAGS.epochs + 1):  # Only 1 training iteration per epoch.
        tic = time()
        inputs = [one_hot, adj_train_norm]

        with tf.GradientTape() as tape:  # Begin gradient calculations from here.
            outputs = model(inputs, training=True)
            weight_decay = sum(tf.reduce_mean(tf.nn.l2_loss(w)) for w in model.weights)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=adj_train_labels, logits=outputs))
            loss += FLAGS.l2 * weight_decay  # Implementing weight decay for further stabilization of training.
        variables = model.trainable_variables
        gradients = tape.gradient(target=loss, sources=variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, variables))

        adjacency_recon = outputs.numpy()  # Slow and inefficient but necessary.
        train_roc, train_ap = get_roc_score(adjacency_recon, real_edges=edges_train, fake_edges=edges_test_fake)
        val_roc, val_ap = get_roc_score(adjacency_recon, real_edges=edges_val, fake_edges=edges_val_fake)

        toc = int(time() - tic)
        # print(f'Epoch {epoch:03d} Time: {toc}s, loss: {float(loss):.3f}')
        tf.summary.scalar(name='Train/Loss', data=loss, step=epoch, description='Training loss for each epoch.')
        tf.summary.scalar(name='Train/ROC', data=train_roc, step=epoch, description='ROC for training set.')
        tf.summary.scalar(name='Train/AP', data=train_ap, step=epoch, description='Average Precision for training set.')
        tf.summary.scalar(name='Val/ROC', data=val_roc, step=epoch, description='ROC for validation set.')
        tf.summary.scalar(name='Val/AP', data=val_ap, step=epoch, description='Average Precision for validation set.')
        print(f'Epoch {epoch} >>> Train Loss: {loss:.2f}, Train ROC: {train_roc:.4f}, Train AP: {train_ap:.4f},'
              f' Val ROC: {val_roc:.4f}, VAL AP: {val_ap:.4f}')

    # After training loop is finished.
    writer.flush()
    print('Training is Finished!')
    test_roc, test_ap = get_roc_score(adjacency_recon, real_edges=edges_test, fake_edges=edges_test_fake)
    print(f'Test ROC: {test_roc:.4f}, Test AP: {test_ap:.4f}')

