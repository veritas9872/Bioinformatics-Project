from pathlib import Path

import numpy as np
import scipy.sparse as sp
import networkx as nx
from absl import flags

FLAGS = flags.FLAGS


def load_edgelist_data(file_path):
    graph = nx.read_edgelist(file_path)
    return nx.adjacency_matrix(graph)


def sparse_to_info(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sp.coo_matrix(sparse_matrix)  # Transform into COO format.
    # Edge coordinates are stored as a long column of pairs. Shape of (N, 2).
    edges = np.vstack([sparse_matrix.row, sparse_matrix.col]).transpose()
    data = sparse_matrix.data
    shape = sparse_matrix.shape
    return edges, data, shape


def process_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])  # Add the identity matrix for inclusion of self features.
    row_sum = np.array(adj_.sum(axis=1))
    # deg is the D\hat^(-1/2) matrix in the paper, the diagonal matrix of the identity augmented adjacency matrix.
    deg = sp.diags((row_sum ** -0.5).ravel())
    adj_norm = sp.coo_matrix(adj_.dot(deg).transpose().dot(deg))
    return sparse_to_info(adj_norm)


def split_graph_edges(sparse_matrix, val_ratio=0.02, test_ratio=0.02):
    """
    Function for building train/validation/test split in graph.
    Also removes diagonal elements. No seeding available.
    """
    sparse_matrix -= sp.diags(sparse_matrix.diagonal())
    sparse_matrix.eliminate_zeros()

    upper_triangular = sp.triu(sparse_matrix)
    upper_triangular_info = sparse_to_info(upper_triangular)
    edge_coords = upper_triangular_info[0]
    edge_coords_all = sparse_to_info(sparse_matrix)[0]
    num_edges = edge_coords_all.shape[0]
    num_val_edges = int(num_edges * val_ratio)
    num_test_edges = int(num_edges * test_ratio)
    num_train_edges = num_edges - num_val_edges - num_test_edges

    edge_idx = np.random.permutation(num_edges)
    train_edges = edge_coords[edge_idx[:num_train_edges]]
    val_edges = edge_coords[edge_idx[num_train_edges:-num_test_edges]]
    test_edges = edge_coords[edge_idx[-num_test_edges:]]

    num_nodes = sparse_matrix.shape[0]

    fake_test_edges = list()
    while len(fake_test_edges) < num_test_edges:
        # Sorting forces the indices to be upper triangular. No replacement forces the indices to be non-diagonal.
        indices = np.random.choice(num_nodes, size=2, replace=False).sort()
        # The "not in" operator can be used because edge_coords is an (N,2) array and indices is a (2,) vector.
        # This allows their shapes to be compatible.
        if (indices not in edge_coords) and (indices not in fake_test_edges):  # Filter out existing edges.
            fake_test_edges.append(indices)

    fake_val_edges= list()
    while len(fake_val_edges) < num_val_edges:
        indices = np.random.choice(num_nodes, size=2, replace=False).sort()
        # Different from SNAP code but similar in effect.
        if (indices not in edge_coords) and (indices not in fake_val_edges):
            fake_val_edges.append(indices)

    # Rebuild the adjacency matrix.
    data = np.ones(shape=num_train_edges)  # Connections are marked by 1.
    # Build a sparse matrix where the locations given by the indices in train_edges
    # are given values of 1 to mark connections.
    # Coordinates in train_edges must be given as separate vectors, hence the horizontal split.
    train_matrix = sp.csr_matrix((data, np.hsplit(train_edges)), shape=sparse_matrix.shape)
    train_matrix += train_matrix.transpose()  # Symmetric matrix should be the same on the transpose.

    return train_matrix, train_edges, val_edges, fake_val_edges, test_edges, fake_test_edges







