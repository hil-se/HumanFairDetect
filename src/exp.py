from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from fermi import FERMI
from load_data import load
import numpy as np
from preprocessor import *
from metrics import Metrics

class Exp:
    def __init__(self, data, treatment="None", inject = None):
        #  Load data
        self.data, self.A = load(data)
        # Separate independent variables and dependent variables
        independent = self.data.keys().tolist()
        dependent = independent.pop(-1)
        self.X = self.data[independent]
        self.y = np.array(self.data[dependent])
        self.treatment = treatment
        self.inject = inject
        if treatment == "FERMI":
            self.clf = FERMI()
        else:
            self.clf = LogisticRegression(max_iter=100000)

    def one_exp(self):
        X_train, X_test, y_train, y_test = self.train_test_split(test_size=0.3)

        #########################################
        self.data_preprocess(X_train)
        #########################################
        y_train_biased = self.inject_bias(X_train, y_train)
        #########################################
        sample_weight = self.treat(X_train, y_train_biased)
        self.fit(X_train, y_train_biased, sample_weight)
        m_test = Metrics(self.clf, X_test, y_test, self.A, self.preprocessor)
        return m_test

    def fit(self, X, y, sample_weight=None):
        X_train_processed = self.preprocessor.fit_transform(X)
        if type(self.clf) == FERMI:
            S = []
            groups = {}
            count = 0
            for i in range(len(y)):
                group = tuple([X[a][i] for a in self.A])
                if group not in groups:
                    groups[group] = count
                    count += 1
                S.append(groups[group])
            S = np.array(S)
            self.clf.fit(X_train_processed, y, S, sample_weight=sample_weight)
        else:
            self.clf.fit(X_train_processed, y, sample_weight=sample_weight)

    def data_preprocess(self, X):
        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(X)
        categorical_columns = categorical_columns_selector(X)

        categorical_preprocessor = OneHotEncoder(handle_unknown = 'ignore')
        numerical_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer([
            ('OneHotEncoder', categorical_preprocessor, categorical_columns),
            ('StandardScaler', numerical_preprocessor, numerical_columns)])

    def treat(self, X_train, y_train):
        if self.treatment == "Reweighing":
            sample_weight = Reweighing(X_train, y_train, self.A)
        elif self.treatment == "FairBalanceVariant":
            sample_weight = FairBalanceVariant(X_train, y_train, self.A)
        elif self.treatment == "ClassBalance":
            sample_weight = ClassBalance(y_train)
        elif self.treatment == "GroupEqual":
            sample_weight = GroupEqual(X_train, self.A)
        elif self.treatment == "FairBalance":
            sample_weight = FairBalance(X_train, y_train, self.A)
        elif self.treatment == "GroupBalanceEqual":
            sample_weight = GroupBalanceEqual(X_train, y_train, self.A)
        else:
            sample_weight = None
        return sample_weight

    def train_test_split(self, test_size=0.5):
        # Split training and testing data proportionally across each group
        groups = {}
        for i in range(len(self.y)):
            key = tuple([self.X[a][i] for a in self.A] + [self.y[i]])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        train = []
        test = []
        for key in groups:
            testing = list(np.random.choice(groups[key], int(len(groups[key])*test_size), replace=False))
            training = list(set(groups[key]) - set(testing))
            test.extend(testing)
            train.extend(training)
        X_train = self.X.iloc[train]
        X_test = self.X.iloc[test]
        y_train = self.y[train]
        y_test = self.y[test]
        X_train.index = range(len(X_train))
        X_test.index = range(len(X_test))
        return X_train, X_test, y_train, y_test

    def inject_bias(self, X_train, y_train):
        if self.inject is not None:
            if self.inject == "synthetic":
                n = len(y_train)
                for i in range(n):
                    hire_rand = np.random.random()
                    hire_prob = 1.0 / (1 + np.exp(15.5 + 10 * X_train["sex"][i] - 2.5 * X_train["work_exp"][i]))
                    hire = 1 if hire_rand < hire_prob else 0
                    y_train[i] = hire
            else:
                X_train_trans = self.preprocessor.fit_transform(X_train)
                clf = LogisticRegression()
                clf.fit(X_train_trans, y_train)
                # ind = clf.classes_[0]
                # pred_score = clf.predict_proba(X_train_trans)[:, 1-ind]
                pred_score = clf.decision_function(X_train_trans)
                for a in self.inject:
                    if self.inject[a] > 0:
                        aa = 1
                    else:
                        aa = 0
                    ind = np.where((X_train[a] == aa) & (y_train == 0))[0]
                    to_change = ind[np.argsort(pred_score[ind])[::-1][:int(len(ind) * np.abs(self.inject[a]))]]
                    y_train[to_change] = 1
                    ind = np.where((X_train[a] == 1-aa) & (y_train == 1))[0]
                    to_change = ind[np.argsort(pred_score[ind])[:int(len(ind) * np.abs(self.inject[a]))]]
                    y_train[to_change] = 0
        return y_train