import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import networkx as nx
from absl import flags

FLAGS = flags.FLAGS


def load_edgelist_data(file_path):
    graph = nx.read_edgelist(file_path)
    return nx.adjacency_matrix(graph)


def sparse_to_info(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sp.coo_matrix(sparse_matrix, shape=sparse_matrix.shape)  # Transform into COO format.
    # Edge coordinates are stored as a long column of pairs. Shape of (N, 2).
    edges = np.vstack([sparse_matrix.row, sparse_matrix.col]).transpose()
    data = sparse_matrix.data
    shape = sparse_matrix.shape
    return edges, data, shape


def process_graph(adj):
    adj = sp.coo_matrix(adj, shape=adj.shape, dtype=np.float32)
    adj_ = adj + sp.identity(adj.shape[0], dtype=np.float32)  # Add the identity matrix for inclusion of self features.
    row_sum = np.array(adj_.sum(axis=1))
    # deg is the D\hat^(-1/2) matrix in the paper, the diagonal matrix of the identity augmented adjacency matrix.
    deg = sp.diags((row_sum ** -0.5).ravel())
    # Normalized adjacency matrix.
    adj_norm = sp.coo_matrix(adj_.dot(deg).transpose().dot(deg), shape=adj.shape, dtype=np.float32)
    adj_norm.eliminate_zeros()
    return sparse_to_info(adj_norm)


def split_graph_edges(sparse_matrix, val_ratio=0.02, test_ratio=0.02, seed=None):
    """
    Function for building train/validation/test split in graph.
    Also removes diagonal elements. No seeding available.
    Maintain symmetry in splitting.
    """
    np.random.seed(seed)  # For reproducibility of data split, etc.

    # Removing diagonal elements.
    sparse_matrix -= sp.diags(sparse_matrix.diagonal(), shape=sparse_matrix.shape, dtype=np.float32)
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
    train_edges = edge_coords[edge_idx[:num_train_edges]].astype(dtype=np.float32)
    val_edges = edge_coords[edge_idx[num_train_edges:-num_test_edges]].astype(dtype=np.float32)
    test_edges = edge_coords[edge_idx[-num_test_edges:]].astype(dtype=np.float32)

    num_nodes = upper_triangular.shape[0]
    edge_coordinates = set(tuple(coord) for coord in edge_coords)

    def make_fake_edges(num_fake_edges):
        fake_edges = edge_coordinates.copy()
        while len(fake_edges) < num_edges + num_fake_edges:
            # Sorting forces the indices to be upper triangular. No replacement forces the indices to be non-diagonal.
            indices = tuple(np.sort(np.random.choice(num_nodes, size=2, replace=False), axis=-1))
            fake_edges.add(indices)  # Efficiently adds only unique indices.
        return np.array(list(fake_edges - edge_coordinates))  # Remove original coordinates.

    fake_test_edges = make_fake_edges(num_test_edges)
    fake_val_edges = make_fake_edges(num_val_edges)

    # Rebuild the adjacency matrix.
    data = np.ones(shape=num_train_edges, dtype=np.float32)  # Connections are marked by 1.
    # Build a sparse matrix where the locations given by the indices in train_edges
    # are given values of 1 to mark connections.
    # Coordinates in train_edges must be given as separate vectors, hence the horizontal split.
    train_matrix = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])),
                                 shape=sparse_matrix.shape, dtype=np.float32)
    # Symmetric matrix should be the same on the transpose.
    train_matrix += train_matrix.transpose()

    # Returning the edge coordinates to symmetric form.
    train_edges = np.concatenate([train_edges, np.fliplr(train_edges)], axis=0).astype(dtype=np.float32)
    val_edges = np.concatenate([val_edges, np.fliplr(val_edges)], axis=0).astype(dtype=np.float32)
    test_edges = np.concatenate([test_edges, np.fliplr(test_edges)], axis=0).astype(dtype=np.float32)

    return train_matrix, train_edges, val_edges, fake_val_edges, test_edges, fake_test_edges


def get_roc_score(adjacency_original, real_edges, fake_edges):

    # adjacency_recon = tf.matmul(embedding, embedding, transpose_b=True)
    real_idx = ((edge[0], edge[1]) for edge in real_edges)
    fake_idx = ((edge[0], edge[1]) for edge in fake_edges)

    # real_recons = np.stack([adjacency_recon[idx] for idx in real_idx], axis=0)
    real_preds = np.exp(np.stack([adjacency_original[idx] for idx in real_idx], axis=0))
    real_preds /= (real_preds + 1)  # Sigmoid (e^x/(e^x + 1)).

    # fake_recons = np.stack([adjacency_recon[idx] for idx in fake_idx], axis=0)
    fake_preds = np.exp(np.stack([adjacency_original[idx] for idx in fake_idx], axis=0))
    fake_preds /= (fake_preds + 1)  # Sigmoid (e^x/(e^x + 1)).

    all_preds = np.concatenate([real_preds, fake_preds], axis=0)
    all_labels = np.concatenate([np.ones(len(real_preds)), np.zeros(len(fake_preds))], axis=0)
    roc_score = roc_auc_score(y_true=all_labels, y_score=all_preds)
    ap_score = average_precision_score(y_true=all_labels, y_score=all_preds)

    return roc_score, ap_score
