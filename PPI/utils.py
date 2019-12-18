import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import networkx as nx
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS


def load_edgelist_data(file_path: str):
    graph = nx.read_edgelist(file_path)
    return nx.adjacency_matrix(graph)


def sparse_to_info(sparse_matrix: sp.spmatrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sp.coo_matrix(sparse_matrix, shape=sparse_matrix.shape)  # Transform into COO format.
    # Edge coordinates are stored as a long column of pairs. Shape of (N, 2).
    edges = np.vstack([sparse_matrix.row, sparse_matrix.col]).transpose()
    data = sparse_matrix.data
    shape = sparse_matrix.shape
    return edges, data, shape


def to_sparse_tensor(sparse_matrix):
    edges, data, shape = sparse_to_info(sparse_matrix)
    return tf.SparseTensor(indices=edges, values=data, dense_shape=shape)


def to_normalized_sparse_tensor(sparse_matrix: sp.spmatrix):
    sparse_matrix = sp.coo_matrix(sparse_matrix, shape=sparse_matrix.shape, dtype=np.float32)
    # Add the identity matrix for inclusion of self features.
    sparse_matrix += sp.identity(sparse_matrix.shape[0], dtype=np.float32)  # Data type conversion is important.
    row_sum = np.array(sparse_matrix.sum(axis=1))
    # deg is the D\hat^(-1/2) matrix in the paper, the diagonal matrix of the identity augmented adjacency matrix.
    deg = sp.diags((row_sum ** -0.5).ravel())
    # Normalized adjacency matrix.
    adj_norm = sp.coo_matrix(sparse_matrix.dot(deg).transpose().dot(deg), shape=sparse_matrix.shape, dtype=np.float32)
    adj_norm.eliminate_zeros()
    return to_sparse_tensor(adj_norm)


def split_graph_edges(sparse_matrix: sp.spmatrix, val_ratio=0.02, test_ratio=0.02, seed=None):
    """
    Function for building train/validation/test split in graph.
    Also removes diagonal elements. No seeding available.
    Maintain symmetry in splitting.
    """
    np.random.seed(seed)  # For reproducibility of data split, etc.

    # Removing diagonal elements.
    sparse_matrix -= sp.diags(sparse_matrix.diagonal(), shape=sparse_matrix.shape)
    sparse_matrix.eliminate_zeros()

    upper_triangular = sp.triu(sparse_matrix)
    upper_triangular_info = sparse_to_info(upper_triangular)
    edge_coords = upper_triangular_info[0]
    # edge_coords_all = sparse_to_info(sparse_matrix)[0]
    num_edges = edge_coords.shape[0]

    num_val_edges = int(num_edges * val_ratio)
    num_test_edges = int(num_edges * test_ratio)
    num_train_edges = num_edges - num_val_edges - num_test_edges

    edge_idx = np.random.permutation(num_edges)
    train_edges = edge_coords[edge_idx[:num_train_edges]]
    val_edges = edge_coords[edge_idx[num_train_edges:-num_test_edges]]
    test_edges = edge_coords[edge_idx[-num_test_edges:]]

    num_nodes = upper_triangular.shape[0]
    edge_coordinates = set(tuple(coord) for coord in edge_coords)

    # Generate fake edges for testing later.
    fake_test_edges = edge_coordinates.copy()  # Use sets for fast removal of duplicates.
    while len(fake_test_edges) < (num_edges + num_test_edges):
        # Indices are always upper triangular when sorted. Not allowing replacement removes diagonal elements.
        indices = tuple(np.sort(np.random.choice(num_nodes, size=2, replace=False), axis=-1))
        fake_test_edges.add(indices)  # Numpy arrays cannot be hashed, hence the need to convert to tuples.

    # Generate fake edges for validation for later.
    fake_val_edges = fake_test_edges.copy()  # Deep copy necessary to prevent adding to original set.
    while len(fake_val_edges) < (num_edges + num_test_edges + num_val_edges):
        indices = tuple(np.sort(np.random.choice(num_nodes, size=2, replace=False), axis=-1))
        fake_val_edges.add(indices)

    # Turn the set into an array of shape (n,2), removing the unnecessary elements from each set.
    fake_val_edges = np.array(list(fake_val_edges - fake_test_edges))
    fake_test_edges = np.array(list(fake_test_edges - edge_coordinates))

    # Rebuild the adjacency matrix.
    data = np.ones(shape=num_train_edges)  # Connections are marked by 1.
    # Build a sparse matrix where the locations given by the indices in train_edges
    # are given values of 1 to mark connections.
    # Coordinates in train_edges must be given as separate vectors, hence the horizontal split.
    train_matrix = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])),
                                 shape=sparse_matrix.shape)
    # Symmetric matrix should be the same on the transpose.
    train_matrix += train_matrix.transpose()

    # Returning the edge coordinates to symmetric form.
    train_edges = np.concatenate([train_edges, np.fliplr(train_edges)], axis=0)
    val_edges = np.concatenate([val_edges, np.fliplr(val_edges)], axis=0)
    test_edges = np.concatenate([test_edges, np.fliplr(test_edges)], axis=0)

    return train_matrix, train_edges, val_edges, fake_val_edges, test_edges, fake_test_edges


def get_roc_score(adjacency_recon: np.ndarray, real_edges, fake_edges):

    real_preds = np.stack([adjacency_recon[edge[0], edge[1]] for edge in real_edges], axis=0)
    real_preds = (1 / (1 + np.exp(-real_preds)))  # Sigmoid. (1 / (1 + e^-x)).

    fake_preds = np.stack([adjacency_recon[edge[0], edge[1]] for edge in fake_edges], axis=0)
    fake_preds = (1 / (1 + np.exp(-fake_preds)))  # Sigmoid. (1 / (1 + e^-x)).

    all_preds = np.concatenate([real_preds, fake_preds], axis=0)
    all_labels = np.concatenate([np.ones(len(real_preds)), np.zeros(len(fake_preds))], axis=0)

    roc_score = roc_auc_score(y_true=all_labels, y_score=all_preds)
    ap_score = average_precision_score(y_true=all_labels, y_score=all_preds)

    return roc_score, ap_score
