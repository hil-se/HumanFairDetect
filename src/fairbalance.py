import copy
from collections import Counter

# Balance training data
def FairBalance(data, class_balance = False):
    y = data.labels.ravel()
    grouping = {}
    for i, label in enumerate(y):
        key = tuple(list(data.protected_attributes[i])+[label])
        if key not in grouping:
            grouping[key]=[]
        grouping[key].append(i)
    class_weight = Counter(y)
    if class_balance:
        class_weight = {key: 1.0 for key in class_weight}
    weighted_data = copy.deepcopy(data)
    for key in grouping:
        weight = class_weight[key[-1]]/len(grouping[key])
        for i in grouping[key]:
            weighted_data.instance_weights[i] = weight
    # Rescale the total weights to len(y)
    weighted_data.instance_weights = weighted_data.instance_weights * len(y) / sum(weighted_data.instance_weights)
    return weighted_data

