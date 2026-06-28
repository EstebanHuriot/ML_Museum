import pandas as pd
import numpy as np

class LogisticRegression:
    
    def __init__(self, epsilon, learning_rate=None, n_iterations=None):
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.epsilon = 1e-15

    def prepare_X(self, X):

        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be 1D or 2D array")
        
        X_b = np.c_[np.ones((len(X), 1)),X]

        return X_b
    
    def prepare_y(self, y):

        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.ndim != 2 or y.shape[1] != 1: # no multiple outputs pls
            raise ValueError("y must be a 1D or 2D array")
        
        return y


    def logistic(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        X_b = self.prepare_X(X)
        y_b = self.prepare_y(y)
        
        # batch gradient descent
        theta = np.random.randn(X_b.shape[1], 1)
        cost_history = []
        m = len(X)

        for iteration in range(self.n_iterations):
    
            p_pred = self.logistic(X_b @ theta)

            cost = -(1 / m) * np.sum(y * np.log(p_pred + self.epsilon) + (1 - y) * np.log(1- p_pred + self.epsilon))
            cost_history.append(cost)

            gradient = (1 / m) * X_b.T @ (p_pred - y)

            theta = theta - self.learning_rate * gradient

        self.theta = theta

        return self
    
    def predict(self, X):

        if self.theta is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_b = self.prepare_X(X)
        
        theta = self.theta
        result = X_b @ theta
        return result

