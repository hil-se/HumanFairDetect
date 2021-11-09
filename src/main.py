import copy
try:
   import cPickle as pickle
except:
   import pickle
from utils import merge_dict, median_dict
from experiment import Experiment
import pandas as pd



def exp_injection1(data = "adult", algorithm = "LR", balance = "FairBalanceClass", repeats=50):
    inject_place = "Train"
    amounts = [0.1, 0.2, 0.3, 0.4]
    results = []
    attr_map = {"age": ['Old', 'Young'], "sex": ['Male', "Female"], "race": ["White", "Non-white"]}
    inject_ratio = {}
    result = exp_injection(algorithm, data, "None", inject_place, inject_ratio, repeats)
    result["Favor"] = "None"
    result["Preprocessing"] = "None"
    results.append(result)
    result = exp_injection(algorithm, data, balance, inject_place, inject_ratio, repeats)
    result["Favor"] = "None"
    result["Preprocessing"] = balance
    results.append(result)
    for amount in amounts:
        if data in {"adult", "german", "compas"}:
            first = 'sex'
            if data in {"german"}:
                second = "age"
            else:
                second = "race"

            inject_ratio = {first: [amount, -amount]}
            result = exp_injection(algorithm, data, balance, inject_place, inject_ratio, repeats)
            result["Favor"] = "%s (%.1f)" %(attr_map[first][1], amount)
            result["Preprocessing"] = balance
            results.append(result)
            inject_ratio = {first: [-amount, amount]}
            result = exp_injection(algorithm, data, balance, inject_place, inject_ratio, repeats)
            result["Favor"] = "%s (%.1f)" %(attr_map[first][0], amount)
            result["Preprocessing"] = balance
            results.append(result)
            inject_ratio = {second: [amount, -amount]}
            result = exp_injection(algorithm, data, balance, inject_place, inject_ratio, repeats)
            result["Favor"] = "%s (%.1f)" %(attr_map[second][1], amount)
            result["Preprocessing"] = balance
            results.append(result)
            inject_ratio = {second: [-amount, amount]}
            result = exp_injection(algorithm, data, balance, inject_place, inject_ratio, repeats)
            result["Favor"] = "%s (%.1f)" %(attr_map[second][0], amount)
            result["Preprocessing"] = balance
            results.append(result)
            inject_ratio = {first: [amount, -amount], second: [amount, -amount]}
            result = exp_injection(algorithm, data, balance, inject_place, inject_ratio, repeats)
            result["Favor"] = "%s (%.1f), %s (%.1f)" % (attr_map[first][1], amount, attr_map[second][1], amount)
            result["Preprocessing"] = balance
            results.append(result)
        else:
            target = "age"
            inject_ratio = {target: [amount, -amount]}
            result = exp_injection(algorithm, data, balance, inject_place, inject_ratio, repeats)
            result["Favor"] = "%s (%.1f)" %(attr_map[target][1], amount)
            result["Preprocessing"] = balance
            results.append(result)
            inject_ratio = {target: [-amount, amount]}
            result = exp_injection(algorithm, data, balance, inject_place, inject_ratio, repeats)
            result["Favor"] = "%s (%.1f)" % (attr_map[target][0], amount)
            result["Preprocessing"] = balance
            results.append(result)
    pd.DataFrame(results).to_csv("../results/bias_injection_"+data+".csv", index=False)



def exp_injection(treatment, data, fair_balance, inject_place, inject_ratio, repeats=10):
    # Conduct one experiment:
    #     treatment in {"SVM", "RF", "LR", "DT"}
    #     data in {"compas", "adult", "german"}
    #     fair_balance in {"None", "FairBalance", "FairBalanceClass"}
    #     inject_place in {"None", "All", "Train"}
    #     inject_ratio={attribute1: [ratio11, ratio12], attribute2: [ratio21, ratio22], ...}
    #     repeats = number of times repeating the experiments

    exp = Experiment(treatment, data=data, fair_balance=fair_balance)
    exp.inject_bias(inject_place, inject_ratio)
    results = {}
    for _ in range(repeats):
        result = exp.run()
        if result:
            results = merge_dict(results, result)
    # print(results)
    medians = copy.deepcopy(results)
    medians = median_dict(medians, use_iqr= True)

    protected = ["sex", "race", "age"]
    for p in protected:
        if p in medians:
            for x in medians[p]:
                medians[p+": "+x] = medians[p][x]
            medians.pop(p)
    print(medians)
    return medians

if __name__ == "__main__":
    for data in ['compas', 'adult', 'heart', 'bank']:
        exp_injection1(data)