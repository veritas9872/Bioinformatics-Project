import deepchem as dc
from rdkit import Chem
import numpy as np
import pickle
import pandas as pd

# m = Chem.MolFromSmiles('COC(=O)CC(NC(=O)c1cnc(NC(=O)OCC2c3ccccc3-c3ccccc32)s1)C(=O)O')#
# # m = Chem.MolFromSmiles('COCCCF')#
# featurizer = dc.feat.ConvMolFeaturizer()#
# features = featurizer.featurize([m])[0]#
# atom_features = features.get_atom_features()  # initial atom feature vectors#
# print(len(atom_features), atom_features[0].shape)#
# adj_list = features.get_adjacency_list()  # adjacency list (neighbor list)#
# print(adj_list)#
# adj=np.zeros((len(adj_list), len(adj_list))) # convert adjacency list into adjacency matrix "A"#
# for i in range(len(adj_list)):#
#     for j in adj_list[i]:#
#         adj[i][j] = 1#
# print(adj.shape)


def adj_list_to_adj_matrix(adj_list: list):
    matrix = np.zeros(shape=(len(adj_list), len(adj_list)), dtype=bool)  # Minimizing data storage requirements.
    for row in range(len(adj_list)):
        for col in adj_list[row]:
            matrix[row, col] = True
    return matrix


if __name__ == '__main__':
    # read_hiv_data('../data/HIV.csv')
    hiv_df = pd.read_csv('../data/HIV.csv')
    smiles = hiv_df['smiles']
    active = hiv_df['HIV_active']

    print('Beginning transformation to molecules.')
    featurizer = dc.feat.ConvMolFeaturizer()
    molecules = [Chem.MolFromSmiles(mol) for mol in smiles]

    print('Beginning featurization process.')
    molecule_features = featurizer.featurize(molecules)
    atom_features = [feature.get_atom_features() for feature in molecule_features]

    adjacency_lists = [feature.get_adjacency_list() for feature in molecule_features]
    adjacency_matrices = [adj_list_to_adj_matrix(adj_list) for adj_list in adjacency_lists]

    activity_labels = np.array(active, dtype=bool)
    hiv_data = {'atom_features': atom_features,
                'adjacency_matrices': adjacency_matrices, 'activity_labels': activity_labels}

    print('Beginning saving!')
    # Saving data to pickle file for easier data loading later.
    # Also removes the need to perform data pre-processing, which is very time consuming, repeatedly.
    with open('./hiv_data.pickle', mode='xb') as file:
        pickle.dump(obj=hiv_data, file=file, protocol=pickle.HIGHEST_PROTOCOL)

