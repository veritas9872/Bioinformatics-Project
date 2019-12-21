import numpy as np
import deepchem as dc
import pandas as pd
from rdkit import Chem
            
def process_prediction(y_true, y_pred):
    """
    1. reshape prediction
    2. remove padding
    """
    result = np.stack(y_pred, axis=1) # reshape prediction
    return result[:len(y_true)] #remove padding            

def make_feature(data_name, feature_name):
    #define featurizer
    if(feature_name == "GraphConv"):
        featurizer = dc.feat.ConvMolFeaturizer()
    elif(feature_name == "ECFP"):
        featurizer = dc.feat.CircularFingerprint(size=1024)    
    
    #define data path
    if(data_name == "HIV"):
        data_path = '../data/HIV.csv'
        active_field = ["HIV_active"]
        smiles_field = "smiles"
        shard_size = 
    elif(data_name == "BACE"):
        data_path = '../data/bace.csv'
        active_field = ["Class"]
        smiles_field = "mol"
    
    # preprocess dataset
    loader = dc.data.CSVLoader(tasks=active_field, smiles_field=smiles_field, featurizer=featurizer) # load data
    feature = loader.featurize(data_path, shard_size=8192) # featurization
    transformer = dc.trans.BalancingTransformer(transform_w=True, dataset=feature) # define transformer    
    feature = transformer.transform(feature) # transformation
    
    return feature
    
def split_data(feature):
    splitter = dc.splits.RandomSplitter() #define splitter
    feature = splitter.train_valid_test_split(feature) # split features
    
    return feature