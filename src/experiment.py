import numpy
import sys
sys.path.append("../")
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import Counter
from fairbalance import FairBalance
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

class Experiment:
    def __init__(self, model, data="adult", fair_balance = "none", target_attribute=""):
        models = {"SVM": LinearSVC(dual=False),
                  "RF": RandomForestClassifier(n_estimators=100, criterion="entropy"),
                  "LR": LogisticRegression(max_iter=1000),
                  "DT": DecisionTreeClassifier(criterion="entropy"),
                  "NB": GaussianNB()
                  }
        data_loader = {"compas": load_preproc_data_compas, "adult": load_preproc_data_adult, "bank": load_preproc_data_bank, "heart": load_preproc_data_heart}

        self.model = models[model]
        self.fair_balance = fair_balance

        # No effect on FairBalance
        self.target_attribute = target_attribute

        self.data = data_loader[data]()
        self.X = self.data.features
        self.y = self.data.labels.ravel()
        self.inject_place = "None"
        self.inject_ratio = None
        self.inject_amount = None
        self.injected = []

    def inject_bias(self, inject_place, inject_ratio):
        # inject_place = {"All": inject to both training and test data, "Train": only inject bias in training data, "None": no biased labels}
        # inject_ratio={attribute1: [ratio11, ratio12], attribute2: [ratio21, ratio22], ...}
        # ratio11 = 0.2 , ratio12 = -0.1 then
        #   1. 20% of (attribute1 =0 AND label = 0) will be changed to label = 1,
        #   2. 10% of (attribute1 =1 AND label = 1) will be changed to label = 0.
        # ratio21 = -0.1 , ratio22 = 0.2 then
        #   1. 10% of (attribute2 =0 AND label = 1) will be changed to label = 0,
        #   2. 20% of (attribute1 =1 AND label = 0) will be changed to label = 1.
        self.inject_place = inject_place
        self.inject_ratio = inject_ratio

    def inject(self, data):
        self.injected = []
        # perform bias injection on the input data.
        if self.inject_ratio:
            for attribute in self.inject_ratio:
                y = data.labels.ravel()
                target = max(y)
                non_target = min(y)
                try:
                    ind = data.protected_attribute_names.index(attribute)
                except:
                    print("Error: Attribute %s does not exist in the protected attributes." %attribute)
                    sys.exit(1)
                groups = data.protected_attributes[:,ind]
                for group, ratio in enumerate(self.inject_ratio[attribute]):
                    to_change = non_target if ratio>0 else target
                    change_to = target if ratio>0 else non_target
                    change = numpy.where((groups == group) & (y==to_change))[0]
                    size = int(numpy.abs(ratio)*len(change))
                    selected = numpy.random.choice(change, size, replace=False)
                    self.injected.extend(list(selected))
                    for i in selected:
                        data.labels[i][0] = change_to
        return data


    def data_prepare(self):
        data_train, data_test = self.data.split([0.7], shuffle=True)
        if self.inject_place!="None":
            data_train = self.inject(data_train)
        if self.inject_place == "All":
            data_test = self.inject(data_test)
        return data_train, data_test

    def run(self):
        data_train, data_test = self.data_prepare()
        if self.fair_balance=="FairBalance":
            dataset_transf_train = FairBalance(data_train, class_balance=False)
        elif self.fair_balance=="FairBalanceClass":
            dataset_transf_train = FairBalance(data_train, class_balance=True)
        else:
            dataset_transf_train = data_train



        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(dataset_transf_train.features_df)
        categorical_columns = categorical_columns_selector(dataset_transf_train.features_df)

        categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
        numerical_preprocessor = StandardScaler()
        preprocessor = ColumnTransformer([
            ('one-hot-encoder', categorical_preprocessor, categorical_columns),
            ('standard-scaler', numerical_preprocessor, numerical_columns)])

        X_train = preprocessor.fit_transform(dataset_transf_train.features_df)
        y_train = dataset_transf_train.labels.ravel()

        self.model.fit(X_train, y_train, sample_weight=dataset_transf_train.instance_weights)

        X_test = preprocessor.transform(data_test.features_df)
        preds = self.model.predict(X_test)

        y_test = data_test.labels.ravel()
        result = self.evaluate(numpy.array(preds), y_test, data_test)
        return result

    def evaluate(self, preds, truth, X_test):
        def rate(a, b):
            aa = Counter(a)[True]
            bb = Counter(b)[True]
            if aa+bb == 0:
                return 0
            else:
                return aa / float(aa+bb)

        result = {}
        # Get target label (for calculating the confusion matrix)
        target = max(set(self.y))
        pp = preds == target
        np = preds != target
        pg = truth == target
        ng = truth != target
        tp = pp & pg
        fp = pp & ng
        tn = np & ng
        fn = np & pg
        result["tpr"] = rate(tp, fn)
        result["fpr"] = rate(fp, tn)
        result["prec"] = rate(tp, fp)
        result["acc"] = rate(tp | tn, fp | fn)
        if (result["tpr"]+result["prec"]) == 0:
            result["f1"] = 0
        else:
            result["f1"] = 2*result["tpr"]*result["prec"]/(result["tpr"]+result["prec"])
        for i, key in enumerate(self.data.protected_attribute_names):
            result[key] = {}
            group1 = X_test.protected_attributes[:,i] == 1
            group0 = X_test.protected_attributes[:,i] == 0
            tp1 = tp & group1
            fp1 = fp & group1
            tn1 = tn & group1
            fn1 = fn & group1
            tp0 = tp & group0
            fp0 = fp & group0
            tn0 = tn & group0
            fn0 = fn & group0
            tpr1 = rate(tp1, fn1)
            fpr1 = rate(fp1, tn1)
            tpr0 = rate(tp0, fn0)
            fpr0 = rate(fp0, tn0)
            result[key]["eod"] = tpr0 - tpr1
            result[key]["aod"] = 0.5*(fpr0-fpr1+tpr0-tpr1)
        return result


