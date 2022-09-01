import numpy as np
from scipy.stats import mannwhitneyu

def is_larger(x, y):
    # Check if results in x is significantly larger than those in y.
    # Return int values:
        # 0: not significantly larger
        # 1: larger with small effect size
        # 2: larger with medium effect size
        # 3: larger with large effect size

    # Mann Whitney U test
    if np.array_equal(x, y):
        return 0
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