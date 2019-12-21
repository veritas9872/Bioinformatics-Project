import numpy as np
            
def process_prediction(y_true, y_pred):
    """
    1. reshape prediction
    2. remove padding
    """
    result = np.stack(y_pred, axis=1) # reshape prediction
    return result[:len(y_true)] #remove padding            