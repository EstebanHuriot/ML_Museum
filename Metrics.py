def r2(y, y_predict):
        
    if y.shape != y_predict.shape:
        raise ValueError('y and y_predict must have the same shape')

    y_bar = y.mean()
    SumSquaredErrors = ((y - y_predict)**2).sum()
    SumSquaresTotal = ((y - y_bar)**2).sum()
    
    r2 = 1 - SumSquaredErrors / SumSquaresTotal
    return r2
    

