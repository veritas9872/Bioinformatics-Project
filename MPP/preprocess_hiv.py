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
    hiv_df = pd.read_csv('../data/HIV.csv')
    smiles = hiv_df['smiles']
    active = hiv_df['HIV_active']
    
    print('Beginning transformation to molecules.')
    featurizer = dc.feat.ConvMolFeaturizer()   
    molecules = [Chem.MolFromSmiles(mol) for mol in smiles]

    print('Beginning featurization process. (CONV)')
    molecule_features = featurizer.featurize(molecules)
    atom_features = [feature.get_atom_features() for feature in molecule_features]
    atom_nums = list(map(len, ([mol for mol in atom_features])))
    max_num = max(atom_nums) # calculate maximum number of atom number

    adjacency_lists = [feature.get_adjacency_list() for feature in molecule_features]
    adjacency_matrices = [adj_list_to_adj_matrix(adj_list, max_num) for adj_list in adjacency_lists]
    pad_nums = [max_num - n for n in atom_nums]
    atom_features_padded = [np.pad(adj,((0,pn),(0,0)),'constant',constant_values=(0)) for (adj,pn) in zip(atom_features, pad_nums)]

    activity_labels = np.array(active, dtype=bool)
    hiv_data_conv = {'atom_features': atom_features_padded,
                    'adjacency_matrices': adjacency_matrices_padded, 'activity_labels': activity_labels}
    
    print('Beginning featurization process. (ECFP)')
    ecfp_degree = 2 # can be changed
    ecfp_power = 9 # can be changed
    ecfp_features = [dc.feat.rdkit_grid_featurizer.compute_ecfp_features(mol, ecfp_degree, ecfp_power) for mol in molecules]

    activity_labels = np.array(active, dtype=bool)
    hiv_data_ecfp = {'ecfp_features': ecfp_features, 'activity_labels': activity_labels}    

    print('Beginning saving!')
    # Saving data to pickle file for easier data loading later.
    # Also removes the need to perform data pre-processing, which is very time consuming, repeatedly.
    with open('./hiv_data_conv.pickle', mode='xb') as file:
        pickle.dump(obj=hiv_data_conv, file=file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./hiv_data_ecfp.pickle', mode='xb') as file:
        pickle.dump(obj=hiv_data_ecfp, file=file, protocol=pickle.HIGHEST_PROTOCOL)        
