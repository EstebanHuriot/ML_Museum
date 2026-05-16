import numpy as np
import pandas as pd


def add_bias_column(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def normal_equation(X, y):
    X_b = add_bias_column(X)
    return np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)