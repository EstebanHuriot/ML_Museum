import numpy as np

def r2(y, y_predict):

    '''
    The R² compares a model's squarred errors to the squarred dispersion around of the target around the mean.
    The closer it is to 1 the more the model explains the variation of y.
    '''
        
    if y.shape != y_predict.shape:
        raise ValueError('y and y_predict must have the same shape')

    y_bar = y.mean()
    SumSquaredErrors = ((y - y_predict)**2).sum()
    SumSquaresTotal = ((y - y_bar)**2).sum()
    
    r2 = 1 - SumSquaredErrors / SumSquaresTotal
    return r2
    

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)