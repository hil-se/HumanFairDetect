from sklearn.linear_model import LogisticRegression
import numpy as np

class MinError:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=100000)
        self.thres = 0

    def fit(self, X, y, sample_weight = None):
        self.clf.fit(X, y, sample_weight=sample_weight)
        self.minError(X, y, sample_weight=sample_weight)

    def minError(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = [1.0] * len(y)
        pos_label = list(self.clf.classes_).index(1)
        probs = self.clf.predict_proba(X)[:,pos_label]
        order = np.argsort(probs)
        # Error for predicting everything as 1
        error = sum((1-y)*np.array(sample_weight))
        minerror = error
        self.thres = 0
        for i in order:
            if y[i]==0:
                error -= sample_weight[i]
                if error < minerror:
                    minerror = error
                    self.thres = probs[i]
            else:
                error += sample_weight[i]

    def predict(self, X):
        pos_label = list(self.clf.classes_).index(1)
        probs = self.clf.predict_proba(X)[:, pos_label]
        return np.array([1 if prob > self.thres else 0 for prob in probs])


class MinErrorDiff:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=100000)
        self.thres = 0

    def fit(self, X, y, sample_weight = None):
        self.clf.fit(X, y, sample_weight=sample_weight)
        self.minErrorDiff(X, y, sample_weight=sample_weight)

    def minErrorDiff(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = [1.0] * len(y)
        pos_label = list(self.clf.classes_).index(1)
        probs = self.clf.predict_proba(X)[:,pos_label]
        order = np.argsort(probs)
        # Error for predicting everything as 1
        FP = sum((1-y)*np.array(sample_weight))
        FN = 0
        minerrordiff = np.abs(FP-FN)
        self.thres = 0
        for i in order:
            if y[i]==0:
                FP -= sample_weight[i]
            else:
                FN += sample_weight[i]
            errordiff = np.abs(FP-FN)
            if errordiff < minerrordiff:
                minerrordiff = errordiff
                self.thres = probs[i]

    def predict(self, X):
        pos_label = list(self.clf.classes_).index(1)
        probs = self.clf.predict_proba(X)[:, pos_label]
        return np.array([1 if prob > self.thres else 0 for prob in probs])