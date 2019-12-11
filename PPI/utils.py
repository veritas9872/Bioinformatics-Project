from pathlib import Path

import numpy as np
import scipy.sparse as sp
import networkx as nx
from absl import flags

FLAGS = flags.FLAGS


def load_edgelist_data(file_path):
    graph = nx.read_edgelist(file_path)
    return nx.adjacency_matrix(graph)



