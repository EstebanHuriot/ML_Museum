import pandas as pd
import numpy as np

class LinearRegression:
    
    def __init__(self, solver, learning_rate, n_iterations):
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        
        X_b = np.c_[np.ones((len(X), 1)),X]
        y_b = np.reshape(y, (len(y), 1))
        
        # normal equation
        if self.solver == 'ne':
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_b
            return theta
        
        # batch gradient descent
        if self.solver  == 'bgd':  
            theta = np.random.randn(2, 1) 

            m = len(X)
            for iteration in range(self.n_iterations):
                gradient = 2 / m * X_b.T @ (X_b @ theta - y)
                theta = theta - gradient * self.learning_rate

            return theta






        
X = np.random.randint(1, 100, size=100)
y = 4 * X

LinearRegression(solver='ne', learning_rate=0.05, n_iterations=100).fit(X, y)

        
    