import numpy as np

# need to add hyperparameters and mostly    min_sampple_leaf
# need to add a get depth method
# need to add a print tree method

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




class BaseDecisionTree:
# Only Gini, no entropy atm
    def __init__(self, max_depth = 5, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None    


    def impurity(self, y):
        raise NotImplementedError("Subclasses must implement the impurity method.") # different classes will use different impurity functions such as gini or mse

    def leaf_value(self, y):
        raise NotImplementedError("Subclasses must implement the impurity method.") # different methods to get leaf value wether its a regression or classifier


    def CostFunction(self, threshold, X: np.ndarray, y: np.ndarray):

        left = X <= threshold   
        right = X > threshold

        G_left = self.impurity(y[left]) 
        G_right = self.impurity(y[right]) 

        m = len(y)
        m_left = len(y[left]) 
        m_right = len(y[right]) 

        J = (m_left/m) * G_left + (m_right/m) * G_right # cost function of the threshold cf the notebook

        return J

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


    def FindBestSplit(self, X, y):

        best_feature= None
        best_threshold = None
        best_cost = np.inf

        
        for feature_idx in range(X.shape[1]):
            feature_values = X[:,feature_idx]

            if len(np.unique(feature_values)) < 2:
                continue

            threshold, cost = self.FindBestThreshold(feature_values, y)

            if cost < best_cost:
                best_feature = feature_idx
                best_threshold = threshold
                best_cost = cost

        return best_feature, best_threshold, best_cost


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

        ):
            return Node(value=self.leaf_value(y)) # return leaf value

        feature, threshold, cost = self.FindBestSplit(X, y)

        if feature is None or threshold is None:
            return Node(value=self.MostCommonClass(y))

        feature_values = X[:, feature]

        left_mask = feature_values <= threshold
        right_mask = feature_values > threshold

        left_node = self.MakeNode(X = X[left_mask], y = y[left_mask], depth = depth + 1)
        right_node = self.MakeNode(X = X[right_mask], y = y[right_mask], depth = depth + 1)

        return Node(feature, threshold=threshold, left=left_node, right=right_node) # return node data


    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1: # accepts (x,) shape
            X = X.reshape(-1, 1)

        if y.ndim != 1:
            raise ValueError(f"y must have shape (n_samples,), but received {y.shape}.")

        if len(X) != len(y): 
            raise ValueError("X and y must contain the same number of samples.")

        self.root = self.MakeNode(X, y, depth=0)     

        return self


    def predictOne(self, x):

        node = self.root

        while node.value is None:

            feature_value = x[node.feature]

            if feature_value <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value


    def predict(self, X):

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = [self.predictOne(x) for x in X]

        return np.array(predictions)



    def print_node(self, node,  depth = 0):

        if node is None:
            return

        indentation = "|" + depth * "-"

        # leaf
        if node.value is not None:
            print(f'{indentation} leaf value:{node.value}')
    
        # decision node
        else:
            print(f'{indentation} decision: k is {node.feature}, t is {node.threshold}')



    def print_tree(self, model):

        if model.root is None:
            print("The tree is empty.")
            return

        def crawler(node, depth=0):

            if node is None:
                return

            self.print_node(node, depth)

            # recursive swag
            if node.value is None:
                crawler(node.left, depth+1)
                crawler(node.right, depth+1)    

        crawler(model.root)




# thank you inheritance
class DecisionTreeClassifier(BaseDecisionTree):

    def impurity(self, y: np.ndarray):

        values, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)

        return 1 - np.sum(proportions ** 2)

    def leaf_value(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]



class DecisionTreeRegressor(BaseDecisionTree):

    def impurity(self, y: np.ndarray):

        mean = np.mean(y)

        return np.mean((y-mean)**2)

    def leaf_value(self, y):
        return np.mean(y)

