from exp import Exp
import pandas as pd
import numpy as np
from stats import is_larger
import time
from demos import cmd

def run(data, repeat = 50, seed = 0):
    np.random.seed(seed)
    treatments = ["None", "FERMI", "Reweighing", "FairBalance", "FairBalanceVariant"]
    metrics = ["Accuracy", "F1 Minority", "EOD", "AOD", "Runtime"]
    columns = ["Treatment"] + metrics
    test_result = {column: [] for column in columns}
    for treat in treatments:
        test_result["Treatment"].append(treat)
        test_accuracy = []
        test_f1 = []
        test_eod = []
        test_aod = []
        test_time = []
        for i in range(repeat):
            t1 = time.time()
            exp = Exp(data = data, treatment = treat)
            m_test = exp.one_exp()
            runtime = time.time() - t1
            test_accuracy.append(m_test.accuracy())
            test_f1.append(m_test.f1_minority())
            test_eod.append(m_test.eod())
            test_aod.append(m_test.aod())
            test_time.append(runtime)
        test_result["Accuracy"].append(test_accuracy)
        test_result["F1 Minority"].append(test_f1)
        test_result["EOD"].append(test_eod)
        test_result["AOD"].append(test_aod)
        test_result["Runtime"].append(test_time)
    for key in metrics:
        if key == "Accuracy" or key == "F1 Minority":
            rank = ranking(test_result[key], better="higher")
        else:
            rank = ranking(test_result[key], better="lower")
        test_result[key] = ["r%d: %.2f (%.2f)" % (rank[i], np.median(test_result[key][i]),
                                                  np.quantile(test_result[key][i], 0.75) - np.quantile(
                                                      test_result[key][i], 0.25)) for i in range(len(test_result[key]))]

    df_test = pd.DataFrame(test_result, columns=columns)
    df_test.to_csv("../results/test/" + data + ".csv", index=False)

def statics(result, better="lower"):
    if better == "higher":
        result = 1.0 - np.array(result)
    medians = [np.median(r) for r in result]
    best = np.argmin(medians)
    rank = []
    for i in range(len(result)):
        if i==best:
            diff = 0
        else:
            diff = is_larger(result[i], result[best])
        rank.append(diff)
    return rank

def ranking(result, better="lower"):
    if better == "higher":
        result = (1.0 - np.array(result)).tolist()
    medians = [np.median(r) for r in result]
    order = np.argsort(medians)
    rankings = [0]*len(order)
    rank = 0
    pre = []
    for i, id in enumerate(order):
        if i==0:
            pre.extend(result[id])
        else:
            diff = is_larger(result[id], pre)
            if diff > 1:
                rank += 1
                pre = result[id]
        rankings[id] = rank
    return rankings

def runAll(seed = 0):
    datasets = ["synthetic1", "synthetic2", "synthetic3", "adult", "compas", "heart", "bank"]
    for data in datasets:
        run(data, seed = seed)

def simple(data = "compas", treat = "FairBalance", inject = {"race": 0.2}, repeats = 30):
    attribute = list(inject.keys())[0]
    result = {"Accuracy": [], "F1 Minority": [], "EOD": [], "AOD": []}
    for i in range(repeats):
        exp = Exp(data=data, treatment=treat, inject=inject)
        m_test = exp.one_exp()
        result["Accuracy"].append(m_test.accuracy())
        result["F1 Minority"].append(m_test.f1_minority())
        result["EOD"].append(m_test.eod(attribute))
        result["AOD"].append(m_test.aod(attribute))
    result = {key: "%d (%d)" % (round(100*np.median(result[key])),
                                    round(100*np.quantile(result[key], 0.75) - np.quantile(
                                        result[key], 0.25))) for key in result}
    return result

def inject_per_data(data = "compas", A = "sex"):
    columns = ["Treatment", "Inject", "Accuracy", "F1 Minority", "EOD", "AOD"]
    results = {key: [] for key in columns}
    result = simple(data=data, treat = "None", inject={A: 0.0})
    results["Treatment"].append("None")
    results["Inject"].append("0")
    for key in result:
        results[key].append(result[key])
    for i in range(5):
        ratio = i / 10
        result = simple(data=data, treat="FairBalance", inject={A: ratio})
        results["Treatment"].append("FairBalance")
        results["Inject"].append("%d" % int(ratio*100))
        for key in result:
            results[key].append(result[key])
    for i in range(1,5):
        ratio = -i / 10
        result = simple(data=data, treat="FairBalance", inject={A: ratio})
        results["Treatment"].append("FairBalance")
        results["Inject"].append("%d" % int(ratio*100))
        for key in result:
            results[key].append(result[key])
    df = pd.DataFrame(results, columns = columns)
    df.to_csv("../inject_results/"+data+"_"+A+".csv", index=False)

def inject_exp_all():
    inject_per_data("adult", "sex")
    inject_per_data("adult", "race")
    inject_per_data("compas", "sex")
    inject_per_data("compas", "race")
    inject_per_data("bank", "age")
    inject_per_data("heart", "age")


if __name__ == "__main__":
    eval(cmd())