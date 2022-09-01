import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class ST:
    def __init__(self, A=["sex"]):
        self.clfs = {}
        self.A = A

    def fit(self, X, y, preprocessor = None):
        self.preprocessor = preprocessor
        if self.preprocessor is None:
            X_train = X.to_numpy()
        else:
            X_train = self.preprocessor.fit_transform(X)
        groups = {}
        for i in range(len(y)):
            key_class = tuple([X[a][i] for a in self.A])
            if key_class not in groups:
                groups[key_class] = []
            groups[key_class].append(i)
        for key in groups:
            self.clfs[key] = LogisticRegression(max_iter=100000, class_weight = "balanced")
            self.clfs[key].fit(X_train[groups[key]], np.array(y)[groups[key]])

    def predict(self, X):
        if self.preprocessor is None:
            X_test = X.to_numpy()
        else:
            X_test = self.preprocessor.transform(X)
        y_pred = []
        for i in range(len(X)):
            key_class = tuple([X[a][i] for a in self.A])
            p = self.clfs[key_class].predict(X_test[i:i+1,:])[0]
            y_pred.append(p)
        return np.array(y_pred)

