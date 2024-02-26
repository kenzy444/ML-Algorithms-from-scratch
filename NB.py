import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
class NaiveBayes :
    def fit(self, X,y):
        n_samples , n_features = X.shape
        # get the number of classes 
        self._classes = np.unique(y)
        n_classses = len(self._classes)

        #claculate mean var and prior for each class 
        self._mean = np.zeros((n_classses, n_features), dtype=np.float64)
        self._prior = np.zeros(n_classses, dtype=np.float64)
        self._var = np.zeros((n_classses, n_features), dtype=np.float64)

        for i ,c in enumerate(self._classes):
            #the X_c is the subset of the data that has the class c
            X_c = X[y ==c]
            self._mean[i,:] = X_c.mean(axis=0)
            self._var[i,:] = X_c.var(axis=0)
            self._prior[i] = X_c.shape[0] /float(n_samples) 


    def predict(self, X):
        y_pred =[self._predict(x) for x in X] 
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors =[]
        # let's calculate posterior proba for each class 
        for i ,c in enumerate(self._classes):
            prior = np.log(self._prior[i]
                           )
            posterior =np.sum(np.log(self._pdf(i,x)))
            posterior += prior
            posteriors.append(posterior) 

        # return the class with the highest posterior
        return self._classes[np.argmax(posteriors)]
    
    # clculate the gaussian
    def _pdf(self, class_inx, x):
        mean = self._mean[class_inx]
        var = self._var[class_inx]
        # review the formula in AssemblyAI video
        numerator = np.exp(-((x-mean)**2)/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator/ denominator
    

  
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.metrics import accuracy_score

    X, y = datasets.make_classification(n_samples=1000,  n_features=10, n_classes=2,random_state=123)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )
    NB = NaiveBayes()
    NB.fit(X_train,y_train)
    predictions = NB.predict(X_test)
    print("Accuracy score is :", accuracy_score(y_test, predictions))
    print("Recall score is :", recall_score(y_test, predictions))
    print("Precision score is :", precision_score(y_test, predictions))



