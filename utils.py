import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

def load_data():
    """
    load data and make adj matrix
    """
    
    graph = nx.read_edgelist('data/yeast.edgelist') # load edge list
    adj_mat = nx.adjacency_matrix(graph) # calculate adj matrix
    return adj_mat

def sparse_matrix_parsing(spmx):
    if not sp.isspmatrix_coo(spmx):
        spmx = sp.coo_matrix(spmx)
    edges = np.vstack((spmx.row, spmx.col)).transpose()
    values = spmx.data
    shape = spmx.shape
    return edges, values, shape  

def sparse2tensor(spmx):
    edges, values, shape = sparse_matrix_parsing(spmx)
    return tf.SparseTensor(edges,values,shape)

def split_data(adj, val_ratio=0.02, test_ratio=0.02):
    """
    1. split data into train / validation / test data.
    2. record negative links for validation / test data.
    3. rebuild adj matrix from train data
    """   
    
    # preprocess adj matrix to extract data of edges
    adj -= sp.diags([adj.diagonal()],[0])
    adj.eliminate_zeros()   
    num_nodes = adj.shape[0]
    adj_triu = sp.triu(adj) # upper triangle of adj matrix (because it is symmetric)
    
    edges = sparse_matrix_parsing(adj_triu)[0]
    all_edges = sparse_matrix_parsing(adj)[0]
    
    # define the number of edges of val / test
    num_edges = edges.shape[0]
    num_val_edges = int(num_edges * val_ratio)
    num_test_edges = int(num_edges * test_ratio)
    
    # sample edges of train / val / test
    edge_idx_shuffled = np.random.permutation(num_edges)
    val_edges = edges[edge_idx_shuffled[:num_val_edges]]
    test_edges = edges[edge_idx_shuffled[-num_test_edges:]]
    train_edges = edges[edge_idx_shuffled[num_val_edges:-num_test_edges]]
    
    print("Data split Success!!")

    # Generate fake edges for testing later.
    edge_coordinates = set(tuple(coord) for coord in edges)
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
            
    # rebuild adj matrix
    rows = train_edges[:,0]
    cols = train_edges[:,1]
    values = np.ones(train_edges.shape[0])    
    train_matrix = sp.csr_matrix((values,(rows,cols)), shape=adj.shape)
    train_matrix += train_matrix.T
                                 
    return train_matrix, train_edges, val_edges, fake_val_edges, test_edges, fake_test_edges, adj

def normalize_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(axis=1))
    inv_sqrt_deg = sp.diags((rowsum ** -0.5).flatten())
    adj_norm = sp.coo_matrix(adj_.dot(inv_sqrt_deg).transpose().dot(inv_sqrt_deg))
    
    return sparse2tensor(adj_norm)

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