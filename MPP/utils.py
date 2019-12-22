import numpy as np
import deepchem as dc


def process_prediction(y_true, y_pred):
    """
    1. reshape prediction
    2. remove padding
    """
    result = np.stack(y_pred, axis=1)  # reshape prediction
    return result[:len(y_true)]  # remove padding


def make_feature(data_name, feature_name):
    # define featurizer
    if feature_name == "GraphConv":
        featurizer = dc.feat.ConvMolFeaturizer()
    elif feature_name == "ECFP":
        featurizer = dc.feat.CircularFingerprint(size=1024)
    else:
        raise ValueError('Invalid feature name!')

        # define data path
    if data_name == "HIV":
        data_path = '../data/HIV.csv'
        active_field = ["HIV_active"]
        smiles_field = "smiles"
    elif data_name == "BACE":
        data_path = '../data/bace.csv'
        active_field = ["Class"]
        smiles_field = "mol"
    else:
        raise ValueError('Invalid data name!')

    # preprocess dataset
    loader = dc.data.CSVLoader(tasks=active_field, smiles_field=smiles_field, featurizer=featurizer, verbose=False)
    feature = loader.featurize(data_path, shard_size=8192)  # featurization
    transformer = dc.trans.BalancingTransformer(transform_w=True, dataset=feature)  # define transformer
    feature = transformer.transform(feature)  # transformation

    return feature


def split_data(feature):
    splitter = dc.splits.RandomSplitter(verbose=False)  # define splitter and split features.
    feature = splitter.train_valid_test_split(feature, frac_train=0.8, frac_valid=0.1, frac_test=0.1, verbose=False)
    return feature
