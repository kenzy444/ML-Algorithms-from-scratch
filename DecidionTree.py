import numpy as np 
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # Constructor for the Node class
        self.feature = feature  # Feature index for the node
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value if the node is a leaf

    def is_leaf_node(self):
        # Check if the node is a leaf node
        return self.value is not None



class DecisionTree():
    def __init__(self, min_samples_split=2 ,n_features = None , max_depth= 1000 ) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features #to add some randomness to the tree we can not use all the features but select only few of them  
        self.root = None 
        
    def fit(self , X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X,y)

    def _grow_tree(self, X, y, depth=0):
        # Recursively grow the decision tree
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self.most_common(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self,X ,y,feat_indxs):
        best_gain = -1
        split_idx , split_threshold = None , None 
        for feat_idx in feat_indxs :
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds :
                gain = self._info_gain(y , X_column, thr)
                if gain > best_gain :
                    best_gain = gain 
                    split_idx = feat_idx
                    split_threshold = thr 
        return split_idx , split_threshold
    
    def _info_gain(self,y , X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)
        # create children 
        left_idx, right_idx= self._split(X_column , threshold )
        if len(left_idx) == 0 or len(right_idx)==0 :
            return 0
        #  calculate the weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l , e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        chiled_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        info_gain = parent_entropy- chiled_entropy 
        return info_gain

    def _split(self, X_column, split_thresh):
        # Split the data into left and right indices based on a threshold
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs



    def _entropy(self,y):
        hist = np.bincount(y)
        ps = hist/ len(y)
        return - np.sum([p* np.log(p) for p in ps if p >0])

    def most_common(self , y ):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value 

    def predict(self , x_test):
        return np.array([self._traverse_tree(x, self.root) for x in x_test])
    
    def _traverse_tree(self, x, node):
        # Recursively traverse the tree to make predictions
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = datasets.load_breast_cancer()
    X , y = data.data , data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    DT = DecisionTree(max_depth=3)
    DT.fit(X_train, y_train) 
    predictions = DT.predict(X_test)
    print("accuracy is :" ,accuracy_score(predictions, y_test) )
