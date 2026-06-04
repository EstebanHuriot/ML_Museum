import pandas as pd
import numpy as np

class LinearRegression:
    
    VALID_SOLVERS = {"ne", "bgd", "sgd"}

    def __init__(self, solver, learning_rate=None, n_iterations=None):
        
        if solver not in self.VALID_SOLVERS:
            raise ValueError(f'{solver} not accepted, solver must be within {self.VALID_SOLVERS}')
        
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.valid_solvers = str(self.VALID_SOLVERS)

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


    def fit(self, X, y):

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        X_b = self.prepare_X(X)
        y_b = self.prepare_y(y)
        
        # normal equation
        if self.solver == 'ne':
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_b
            
            self.theta = theta
            return self
        
        # batch gradient descent
        if self.solver  == 'bgd':  
            theta = np.random.randn(X_b.shape[1], 1)

            m = len(X)
            for iteration in range(self.n_iterations):
                gradient = 2 / m * X_b.T @ (X_b @ theta - y_b)
                theta = theta - gradient * self.learning_rate

            self.theta = theta
            return self

        # stochastic gradient descent
        if self.solver == 'sgd':
            theta = np.random.randn(X_b.shape[1], 1)

            m = len(X)
            for epoch in range(self.n_iterations):
                for i in range(m):
                    random_index = np.random.randint(m)

                    Xi = X_b[random_index:random_index+1]
                    yi = y_b[random_index:random_index+1]

                    gradients = 2 * Xi.T.dot(Xi.dot(theta) - yi)
                    theta = theta - gradients * self.learning_rate
            
            self.theta = theta
            return self
        

    def predict(self, X):

        if self.theta is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_b = self.prepare_X(X)
        
        theta = self.theta
        result = X_b @ theta
        return result


        
    