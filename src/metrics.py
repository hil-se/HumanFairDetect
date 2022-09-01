from collections import Counter
import numpy as np

class Metrics:

    def __init__(self, clf, X_test, y_test, A, preprocessor=None):
        # clf: the classifier/predictor to be evaluated
        # X_test: test data, independent variables (2-d pd.DataFrame)
        # y_test: test data, the dependent variable (1-d np.array, binary y_test in {0,1})
        # preprocessor: the preprocessor for X_test
        self.clf = clf
        self.X = X_test
        self.y = y_test
        self.A = A
        self.preprocessor = preprocessor
        if self.preprocessor:
            self.y_pred = self.clf.predict(self.preprocessor.transform(self.X))
        else:
            self.y_pred = self.clf.predict(self.X)
        self.groups = {}
        for i in range(len(self.y)):
            group = tuple([self.X[a][i] for a in A])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)

    def accuracy(self):
        return Counter(self.y==self.y_pred)[True] / len(self.y)

    def f1_minority(self):
        target1 = Counter(self.y)[1] <= Counter(self.y)[0]
        conf = self.confusion()
        if target1:
            tpr = conf['tp'] / (conf['tp'] + conf['fn'])
            prec = conf['tp'] / (conf['tp'] + conf['fp'])
        else:
            tpr = conf['tn'] / (conf['tn'] + conf['fp'])
            prec = conf['tn'] / (conf['tn'] + conf['fn'])
        f1 = 2*tpr*prec/(tpr+prec)
        return f1

    def tprs(self):
        tprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusion(sub)
            tpr = conf['tp'] / (conf['tp'] + conf['fn'])
            tprs[group] = tpr
        return tprs

    def prs(self):
        prs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusion(sub)
            pr = (conf['tp']+conf['fp']) / len(sub)
            prs[group] = pr
        return prs

    def fprs(self):
        fprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusion(sub)
            fpr = conf['fp'] / (conf['fp'] + conf['tn'])
            fprs[group] = fpr
        return fprs

    def aos(self):
        tprs = self.tprs()
        fprs = self.fprs()
        aos = {key: (tprs[key]+fprs[key])/2 for key in tprs}
        return aos


    def eod(self, a=None):
        # return: EOD = max{TPR(A=a)} - min{TPR(A=a)} if a == None
        # return: EOD = TPR(a=1) - TPR(a=0) if a != None
        # TPR = #(y=1, C=1) / #(y=1)
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.confusion(ind0)
            conf1 = self.confusion(ind1)
            tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
            tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])
            return tpr1 - tpr0
        else:
            tprs = self.tprs()
            return max(tprs.values())-min(tprs.values())

    def aod(self, a=None):
        # return: AOD = 0.5 * (max{TPR(A=a)+FPR(A=a)} - min{TPR(A=a)+FPR(A=a)}) if a == None
        # return: AOD = 0.5 * (TPR(a=1) + FPR(a=1) - TPR(a=0) - FPR(a=0)) if a != None
        # TPR = #(y=1, C=1) / #(y=1), FPR = #(y=0, C=1) / #(y=0)
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.confusion(ind0)
            conf1 = self.confusion(ind1)
            tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
            tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])
            fpr0 = conf0['fp'] / (conf0['fp'] + conf0['tn'])
            fpr1 = conf1['fp'] / (conf1['fp'] + conf1['tn'])
            return 0.5 * (tpr1 + fpr1 - tpr0 - fpr0)
        else:
            aos = self.aos()
            return max(aos.values()) - min(aos.values())

    def spd(self, a=None):
        # return: SPD = max{TPR(A=a)} - min{TPR(A=a)} if a == None
        # return: SPD = PR(a=1) - PR(a=0) if a != None
        # PR = #(C=1) / N
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.confusion(ind0)
            conf1 = self.confusion(ind1)
            pr0 = (conf0['tp']+conf0['fp']) / len(ind0)
            pr1 = (conf1['tp']+conf1['fp']) / len(ind1)
            return pr1 - pr0
        else:
            prs = self.prs()
            return max(prs.values())-min(prs.values())

    def confusion(self, sub=None):
        if sub is None:
            sub = range(len(self.y))
        y = self.y[sub]
        y_pred = self.y_pred[sub]
        conf = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
        for i in range(len(y)):
            if y[i]==0 and y_pred[i]==0:
                conf['tn']+=1
            elif y[i]==1 and y_pred[i]==1:
                conf['tp'] += 1
            elif y[i]==0 and y_pred[i]==1:
                conf['fp'] += 1
            elif y[i]==1 and y_pred[i]==0:
                conf['fn'] += 1
        return conf

    def diff(self, a=None):
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.confusion(ind0)
            conf1 = self.confusion(ind1)
            diff0 = (conf0['fp'] - conf0['fn']) / float(len(ind0))
            diff1 = (conf1['fp'] - conf1['fn']) / float(len(ind1))
            return diff1 - diff0
        else:
            diffs = {}
            for group in self.groups:
                sub = self.groups[group]
                conf = self.confusion(sub)
                diff = (conf['fp'] - conf['fn']) / float(len(sub))
                diffs[group] = diff
            return max(diffs.values()) - min(diffs.values())







