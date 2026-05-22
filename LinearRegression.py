import pandas as pd
import numpy as np

class LinearRegression:
    
    VALID_SOLVERS = {"ne", "bgd", "sgd"}

    def __init__(self, solver, learning_rate=None, n_iterations=None, epochs=None):
        
        if solver not in self.VALID_SOLVERS:
            raise ValueError(f'{solver} not accepted, solver must be within {self.VALID_SOLVERS}')
        
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.epochs = epochs

    def fit(self, X, y):
        
        X_b = np.c_[np.ones((len(X), 1)),X]
        y_b = np.reshape(y, (len(y), 1))
        
        # normal equation
        if self.solver == 'ne':
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_b
            return theta
        
        # batch gradient descent
        if self.solver  == 'bgd':  
            theta = np.random.randn(X_b.shape[1], 1)

            m = len(X)
            for iteration in range(self.n_iterations):
                gradient = 2 / m * X_b.T @ (X_b @ theta - y_b)
                theta = theta - gradient * self.learning_rate

            return theta

        # stochastic gradient descent
        if self.solver == 'sgd':
            theta = np.random.randn(X_b.shape[1], 1)

            m = len(X)
            for epoch in range(self.epochs):
                for i in range(m):
                    random_index = np.random.randint(100)

                    Xi = X_b[random_index:random_index+1]
                    yi = y_b[random_index:random_index+1]

                    gradients = 2 * Xi.T.dot(Xi.dot(theta) - yi)
                    theta = theta - gradients * self.learning_rate
            
            return theta

        
    