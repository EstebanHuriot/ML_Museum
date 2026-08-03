import numpy as np



class Node:
# Both decision and leaf atm
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):

        # decision node
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

        # leaf
        self.value = value









class DecisionTree:
# Only Gini, no entropy atm
    def __init__(self, max_depth = 5, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None    


    def CostFunction(self, threshold, X: np.ndarray, y: np.ndarray):

        left = X <= threshold   
        right = X > threshold

        G_left = self.Gini(y[left]) 
        G_right = self.Gini(y[right]) 

        m = len(y)
        m_left = len(y[left]) 
        m_right = len(y[right]) 

        J = (m_left/m) * G_left + (m_right/m) * G_right # cost function of the threshold cf the notebook
        return J



    def Gini(self, y: np.ndarray):

        values, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)

        return 1 - np.sum(proportions ** 2)


    def FindBestThreshold(self, X, y):

        thresholds = (np.unique(X)[:-1] + np.unique(X)[1:]) / 2 # values in between values
        costs = []

        for threshold in thresholds: # testing every in between values as threshold

            J = self.CostFunction(threshold, X, y)
            costs.append(J)

        cheapest_idx = np.argmin(costs) # lowest cost function's index

        best_t = thresholds[cheapest_idx] # best threshold 
        best_cost = costs[cheapest_idx]       

        return best_t, best_cost


    def MostCommonClass(self, y):

        values, counts = np.unique(y, return_counts=True)
        most_common_index = np.argmax(counts)

        return values[most_common_index]

    



    def MakeNode(self, X, y, depth = 0):

        X = np.array(X)
        y = np.array(y)

        if (
            len(np.unique(y)) == 1 # nothing to divide
        or depth >= self.max_depth # stop at desired depth
        or len(y) < self.min_samples_split # min sample number
        or len(np.unique(X)) < 2 # min number of unique X values

        ):
            value = self.MostCommonClass(y)
            return Node(value=value)


        
        t, cost = self.FindBestThreshold(X, y)

        left_mask = X <= t
        right_mask = X > t

        left_node = self.MakeNode(X = X[left_mask], y = y[left_mask], depth = depth + 1)
        right_node = self.MakeNode(X = X[right_mask], y = y[right_mask], depth = depth + 1)

        return Node(feature=0, threshold=t, left=left_node, right=right_node)




    


