import deepchem as dc
from rdkit import Chem
import numpy as np
import pickle
import pandas as pd

def adj_list_to_adj_matrix(adj_list: list, max_num: int):
    matrix = np.zeros(shape=(max_num, max_num), dtype=bool)  # Minimizing data storage requirements.
    for row in range(len(adj_list)):
        for col in adj_list[row]:
            matrix[row, col] = True
    return matrix

if __name__ == '__main__':
    bace_df = pd.read_csv('../data/BACE.csv')
    smiles = bace_df['mol']
    active = bace_df['Class']

    print('Beginning transformation to molecules.')
    featurizer = dc.feat.ConvMolFeaturizer()
    molecules = [Chem.MolFromSmiles(mol) for mol in smiles]

    print('Beginning featurization process. (CONV)')
    molecule_features = featurizer.featurize(molecules)
    atom_features = [feature.get_atom_features() for feature in molecule_features]
    max_num = max(map(len,([mol for mol in atom_features]))) # calculate maximum number of atom number

    adjacency_lists = [feature.get_adjacency_list() for feature in molecule_features]
    adjacency_matrices = [adj_list_to_adj_matrix(adj_list, max_num) for adj_list in adjacency_lists]

    activity_labels = np.array(active, dtype=bool)
    bace_data_conv = {'atom_features': atom_features,
                    'adjacency_matrices': adjacency_matrices, 'activity_labels': activity_labels}

    print('Beginning featurization process. (ECFP)')
    ecfp_degree = 2 # can be changed
    ecfp_power = 9 # can be changed
    ecfp_features = [dc.feat.rdkit_grid_featurizer.compute_ecfp_features(mol, ecfp_degree, ecfp_power) for mol in molecules]

    activity_labels = np.array(active, dtype=bool)
    bace_data_ecfp = {'ecfp_features': ecfp_features, 'activity_labels': activity_labels}
    
    print('Beginning saving!')
    # Saving data to pickle file for easier data loading later.
    # Also removes the need to perform data pre-processing, which is very time consuming, repeatedly.
    with open('./bace_data_conv.pickle', mode='xb') as file:
        pickle.dump(obj=bace_data_conv, file=file, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('./bace_data_ecfp.pickle', mode='xb') as file:
        pickle.dump(obj=bace_data_ecfp, file=file, protocol=pickle.HIGHEST_PROTOCOL)