from DecidionTree import DecisionTree
import numpy as np 
from collections import Counter

class random_forest():
    def __init__(self, min_samples_split=2, n_features = None , max_depth = 10, n_trees = 10):
        self.min_samples_split = min_samples_split
        self.n_features = n_features 
        self.max_depth = max_depth 
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split ,n_features =self.n_features , max_depth=self.max_depth )
            x_samples , y_samples = self.get_samples( X, y)
            tree.fit(x_samples, y_samples)
            self.trees.append(tree)

    def get_samples(self, X ,y):
        n_samples = X.shape[0]
        indx = np.random.choice(n_samples , n_samples , replace=True)
        return X[indx], y [indx]
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # it will return an array of arrays(it will be an array for each tree in the random forest ie 10 in this case) each array has the predictions of each sample of X so a simple will have a prediction from all trees 
        # make them in one list ie  for each simple [pred_value_from_tree1, pred_value_from_tree2,..... ]
        tree_pred = np.swapaxes(predictions,0,1)
        predictions = [self.most_common(pred) for pred in tree_pred]
        return predictions

    def most_common(self , y ):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value 
    


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = datasets.load_breast_cancer()
    X , y = data.data , data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    RF = random_forest(n_trees=20)
    RF.fit(X_train, y_train) 
    predictions = RF.predict(X_test)
    print("accuracy is :" ,accuracy_score(predictions, y_test) )



