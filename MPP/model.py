import tensorflow as tf
import numpy as np

from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.layers import GraphConv, BatchNorm
from deepchem.models.tensorgraph.layers import GraphPool, GraphGather
from deepchem.models.tensorgraph.layers import Dense, SoftMax, SoftMaxCrossEntropy, WeightedError, ReduceMean
from deepchem.models.tensorgraph.layers import Label, Weights
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.tensor_graph import TensorGraph


class GCN:
    def __init__(self, batch_size=50):
        # save parameters
        self.batch_size = batch_size

        # define tensorgraph
        self.tg = TensorGraph(use_queue=False)

        # define features
        self.atom_features = Feature(shape=(None, 75))  # feature of atom. ex) atom / degree / is aromatic and so on
        self.indexing = Feature(shape=(None, 2), dtype=tf.int32)  # index of atoms in molecules sorted by degree
        self.membership = Feature(shape=(None,), dtype=tf.int32)  # membership of atoms in molecule
        self.deg_adj_list = [Feature(shape=(None, i), dtype=tf.int32) for i in range(1, 12)]  # adj list with degree

        # build graph
        self.build_graph()

    def build_graph(self):
        # Layer 1
        gc1_input = [self.atom_features, self.indexing, self.membership] + self.deg_adj_list
        gc1 = GraphConv(64, activation_fn=tf.nn.relu, in_layers=gc1_input)
        bn1 = BatchNorm(in_layers=[gc1])
        gp1_input = [bn1, self.indexing, self.membership] + self.deg_adj_list
        gp1 = GraphPool(in_layers=gp1_input)

        # Layer 2
        gc2_input = [gp1, self.indexing, self.membership] + self.deg_adj_list
        gc2 = GraphConv(64, activation_fn=tf.nn.relu, in_layers=gc2_input)
        bn2 = BatchNorm(in_layers=[gc2])
        gp2_input = [bn2, self.indexing, self.membership] + self.deg_adj_list
        gp2 = GraphPool(in_layers=gp2_input)

        # Dense layer 1
        d1 = Dense(out_channels=128, activation_fn=tf.nn.relu, in_layers=[gp2])
        bn3 = BatchNorm(in_layers=[d1])

        # Graph gather layer
        gg1_input = [bn3, self.indexing, self.membership] + self.deg_adj_list
        gg1 = GraphGather(batch_size=self.batch_size, activation=tf.nn.tanh, in_layers=gg1_input)

        # Output dense layer
        d2 = Dense(out_channels=2, activation_fn=None, in_layers=[gg1])
        softmax = SoftMax(in_layers=[d2])
        self.tg.add_output(softmax)

        # Set loss function
        self.label = Label(shape=(None, 2))
        cost = SoftMaxCrossEntropy(in_layers=[self.label, d2])
        self.weight = Weights(shape=(None, 1))
        loss = WeightedError(in_layers=[cost, self.weight])
        self.tg.set_loss(loss)

    def fit(self, dataset, epochs:int):
        self.tg.fit_generator(self.data_generator(dataset, self.batch_size, epochs=epochs))

    def predict(self, dataset):
        pred = self.tg.predict_on_generator(self.data_generator(dataset, self.batch_size))
        return np.expand_dims(pred, axis=0)

    def data_generator(self, dataset, batch_size:int, epochs=1):
        for e in range(epochs):
            for X, y, w, idx in dataset.iterbatches(batch_size, pad_batches=True, deterministic=True):
                feed_dict = {self.label: to_one_hot(y[:, 0]), self.weight: w}  # data for feed
                ConvMolList = ConvMol.agglomerate_mols(X)
                feed_dict[self.atom_features] = ConvMolList.get_atom_features()
                feed_dict[self.indexing] = ConvMolList.deg_slice
                feed_dict[self.membership] = ConvMolList.membership
                deg_adj_list = ConvMolList.get_deg_adjacency_lists()
                for i in range(1, len(deg_adj_list)):
                    feed_dict[self.deg_adj_list[i - 1]] = deg_adj_list[i]

                yield feed_dict


class MLP:
    def __init__(self, batch_size):
        # save parameters
        self.batch_size = batch_size

        # define tensorgraph
        self.tg = TensorGraph(use_queue=False)
        self.feature = Feature(shape=(None, 1024))

        # build graph
        self.build_graph()

    def build_graph(self):
        d1 = Dense(out_channels=256, activation_fn=tf.nn.relu, in_layers=[self.feature])
        d2 = Dense(out_channels=64, activation_fn=tf.nn.relu, in_layers=[d1])
        d3 = Dense(out_channels=16, activation=None, in_layers=[d2])
        d4 = Dense(out_channels=2, activation=None, in_layers=[d3])
        softmax = SoftMax(in_layers=[d4])
        self.tg.add_output(softmax)

        self.label = Label(shape=(None, 2))
        cost = SoftMaxCrossEntropy(in_layers=[self.label, d4])
        loss = ReduceMean(in_layers=[cost])
        self.tg.set_loss(loss)

    def fit(self, dataset, epochs):
        self.tg.fit_generator(self.data_generator(dataset, self.batch_size, epochs=epochs))

    def predict(self, dataset):
        pred = self.tg.predict_on_generator(self.data_generator(dataset, self.batch_size))
        return np.expand_dims(pred, axis=0)

    def data_generator(self, dataset, batch_size, epochs=1):
        for e in range(epochs):
            for X, y, w, idx in dataset.iterbatches(batch_size, pad_batches=True, deterministic=True):
                feed_dict = {self.label: to_one_hot(y[:, 0]), self.feature: X}  # data for feed

                yield feed_dict
