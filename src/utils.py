import numpy as np
from scipy.stats import mannwhitneyu
try:
   import cPickle as pickle
except:
   import pickle


def merge_dict(results, result):
    # Merge nested dictionaries
    for key in result:
        if type(result[key]) == dict:
            if key not in results:
                results[key] = {}
            results[key] = merge_dict(results[key], result[key])
        else:
            if key not in results:
                results[key] = []
            results[key].append(result[key])
    return results



def is_larger(x, y):
    # Check if results in x is significantly larger than those in y.
    # Return int values:
        # 0: not significantly larger
        # 1: larger with small effect size
        # 2: larger with medium effect size
        # 3: larger with large effect size

    # Mann Whitney U test
    U, pvalue = mannwhitneyu(x, y, alternative="greater")
    if pvalue>0.05:
        # If x is not greater than y in 95% confidence
        return 0
    else:
        # Calculate Cliff's delta with U
        delta = 2*U/(len(x)*len(y))-1
        # Return different levels of effect size
        if delta<0.147:
            return 0
        elif delta<0.33:
            return 1
        elif delta<0.474:
            return 2
        else:
            return 3

def compare_dict(results, baseline="None"):
    # Check if results of non-baseline treatments are significantly better than the baseline

    y = results[baseline]
    for treatment in results:
        if treatment==baseline:
            continue
        x = results[treatment]
        for key in x:
            if type(x[key]) == dict:
                # Bias Metrics: lower the better
                for key2 in x[key]:
                    xx = x[key][key2]
                    yy = y[key][key2]
                    better = is_larger(np.abs(yy), np.abs(xx))
                    if better == 0:
                        better = -is_larger(np.abs(xx), np.abs(yy))
                    x[key][key2] = better
            else:
                # General Metrics: higher the better
                xx = x[key]
                yy = y[key]
                better = is_larger(xx, yy)
                if better == 0:
                    better = -is_larger(yy, xx)
                x[key] = better
    for key in y:
        if type(y[key]) == dict:
            for key2 in y[key]:
                y[key][key2] = "n/a"
        else:
            y[key] = "n/a"
    return results

def median_dict(results, use_iqr = True):
    # Compute median value of lists in the dictionary
    for key in results:
        if type(results[key]) == dict:
            results[key] = median_dict(results[key], use_iqr = use_iqr)
        else:
            med = np.median(results[key])
            if use_iqr:
                iqr = np.percentile(results[key],75)-np.percentile(results[key],25)
                results[key] = "%d (%d)" % (med*100, iqr*100)
            else:
                results[key] = "%d" % (med*100)
    return results

def mean_dict(results, std = True):
    # Compute mean value of lists in the dictionary
    for key in results:
        if type(results[key]) == dict:
            results[key] = median_dict(results[key])
        else:
            med = np.mean(results[key])
            if std:
                std = np.std(results[key])
                results[key] = "%.2f (%.2f)" % (med, std)
            else:
                results[key] = "%.2f" % (med)
    return results

def color(median, compare):
    mapping = {3: "\\cellcolor{green!70}", 2: "\\cellcolor{green!35}", 1: "\\cellcolor{green!15}",
               -3: "\\cellcolor{red!70}", -2: "\\cellcolor{red!35}", -1: "\\cellcolor{red!15}",
               0: "", "n/a": ""}
    for key in median:
        if type(median[key]) == dict:
            median[key] = color(median[key], compare[key])
        else:
            median[key] = mapping[compare[key]]+median[key]
    return median

def AUC(result):
    tp=0
    fp=0
    auc = 0
    for x in result:
        if x>0:
            tp+=1
        else:
            fp+=1
            auc += tp
    if tp==0:
        return 0.0
    elif fp==0:
        return 1.0
    else:
        return float(auc) / tp / fp

def topK_precision(result, K):
    return np.mean(result[:K])