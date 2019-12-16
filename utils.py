import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp

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
    num_edges_all = edges.shape[0]
    num_edges_val = int(num_edges_all * val_ratio)
    num_edges_test = int(num_edges_all * test_ratio)
    
    # sample edges of train / val / test
    edge_idx_shuffled = np.random.permutation(num_edges_all)
    val_edges = edges[edge_idx_shuffled[:num_edges_val]]
    test_edges = edges[edge_idx_shuffled[-num_edges_test:]]
    train_edges = edges[edge_idx_shuffled[num_edges_val:-num_edges_test]]
    
    train_edges_l = train_edges.tolist()
    val_edges_l = val_edges.tolist()
    test_edges_l = test_edges.tolist()
    all_edges_l = all_edges.tolist()
    
    print("Data split Success!!")
    
    # val_edges_false
    val_edges_false = []
    while len(val_edges_false) < num_edges_val:
        rnd_num = num_edges_val - len(val_edges_false)
        print("remain # of edges : ", rnd_num)             
        rnd = np.random.randint(0, num_nodes, size = 2*rnd_num)
        indices = np.sort(np.stack((rnd[:rnd_num], rnd[rnd_num:]), axis=-1))
        indices_l = indices.tolist()
        
        for idx in indices_l:
            if idx[0] == idx[1]:
                continue
            if (idx in train_edges_l) or (idx[::-1] in train_edges_l):
                continue               
            if (idx in val_edges_l) or (idx[::-1] in val_edges_l):
                continue  
            if (idx in val_edges_false) or (idx[::-1] in val_edges_false):
                continue             
            val_edges_false.append(idx)
                    
    print("val_edge_false generated!!")
    

    # test_edges_false
    test_edges_false = []
    while len(test_edges_false) < num_edges_test:
        rnd_num = num_edges_test - len(test_edges_false)
        print("remain # of edges : ", rnd_num)      
        rnd = np.random.randint(0, num_nodes, size = 2*rnd_num)
        indices = np.sort(np.stack((rnd[:rnd_num], rnd[rnd_num:]), axis=-1))
        indices_l = indices.tolist()
        
        for idx in indices_l:
            if idx[0] == idx[1]:
                continue
            if (idx in test_edges_l) or (idx[::-1] in test_edges_l):
                continue               
            if (idx in all_edges_l):
                continue  
            if (idx in test_edges_false) or (idx[::-1] in test_edges_false):
                continue             
            test_edges_false.append(idx)
            
    print("test_edge_false generated!!")                

            
    # rebuild adj matrix
    rows = train_edges[:,0]
    cols = train_edges[:,1]
    values = np.ones(train_edges.shape[0])    
    train_matrix = sp.csr_matrix((values,(rows,cols)), shape=adj.shape)
    train_matrix += train_matrix.T
                                 
    return train_matrix, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj

def normalize_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(axis=1))
    inv_sqrt_deg = sp.diags((rowsum ** -0.5).flatten())
    adj_norm = sp.coo_matrix(adj_.dot(inv_sqrt_deg).transpose().dot(inv_sqrt_deg))
    
    return sparse2tensor(adj_norm)