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


def mask_graph_edges(adj, val_ratio=0.02, test_ratio=0.02):
    """
    Function for building train/validation/test split in graph.
    Also removes diagonal elements.
    """
    adj -= sp.diags(adj.diagonal())
    adj.eliminate_zeros()

    adj_triu = sp.triu(adj)
    adj_info_triu = sparse_to_info(adj_triu)
    edge_coords = adj_info_triu[0]
    edge_coords_all = sparse_to_info(adj)[0]
    num_edges = edge_coords_all.shape[0]
    num_val_edges = int(num_edges * val_ratio)
    num_test_edges = int(num_edges * test_ratio)
    num_train_edges = num_edges - num_val_edges - num_test_edges

    edge_idx = np.random.permutation(num_edges)
    train_edges = edge_idx[:num_train_edges]
    val_edges = edge_idx[num_train_edges:-num_test_edges]
    test_edges = edge_idx[-num_test_edges:]

    # TODO: Finish this function!
