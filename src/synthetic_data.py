import numpy as np
import pandas as pd
from pdb import set_trace

def make_data(n=5000, p=0.5, l=6):
    # n is the number of data points.
    # 0 <= p <= 1 is the sampling probability of Male (sex=1).
    keys = ["sex", "work_exp", "hair_length", "hire"]
    data = {key: [] for key in keys}
    for i in range(n):
        rand = np.random.random()
        sex = 1 if rand < p else 0
        hair_length = 35 * np.random.beta(2, 2+5*sex)
        # work_exp = int(np.random.poisson(5 + 6 * sex))
        work_exp = int(np.random.poisson(25+l*sex) - np.random.normal(20, 0.2))
        if work_exp < 0:
            work_exp = 0
        hire_rand = np.random.random()
        hire_prob = 1.0/(1+np.exp(25.5-2.5*work_exp))
        # hire_prob = 1.0 / (1 + np.exp(15.5  +10*sex - 2.5 * work_exp))
        # hire_prob = 1.0 / (1 + np.exp(8 - work_exp))
        hire = 1 if hire_rand < hire_prob else 0
        data["sex"].append(sex)
        data["work_exp"].append(work_exp)
        data["hair_length"].append(hair_length)
        data["hire"].append(hire)
    df = pd.DataFrame(data, columns = keys)
    return df

def make_data2(n=5000, p=0.5, l=6):
    # n is the number of data points.
    # 0 <= p <= 1 is the sampling probability of Male (sex=1).
    keys = ["sex", "age", "work_exp", "hair_length", "hire"]
    data = {key: [] for key in keys}
    for i in range(n):
        rand = np.random.random()
        sex = 1 if rand < p else 0
        age = int(np.random.normal(25, 3))
        hair_length = 35 * np.random.beta(2, 2+5*sex)
        work_exp = int(np.random.poisson(6+age-l*(1-sex)) - np.random.normal(20, 0.2))
        if work_exp < 0:
            work_exp = 0
        hire_rand = np.random.random()
        hire_prob = 1.0/(1+np.exp(25.5-2.5*work_exp))
        # hire_prob = 1.0 / (1 + np.exp(15.5  +10*sex - 2.5 * work_exp))
        # hire_prob = 1.0 / (1 + np.exp(8 - work_exp))
        hire = 1 if hire_rand < hire_prob else 0
        data["sex"].append(sex)
        data["age"].append(age)
        data["work_exp"].append(work_exp)
        data["hair_length"].append(hair_length)
        data["hire"].append(hire)
    df = pd.DataFrame(data, columns = keys)
    return df

def make_data3(n=5000, p=0.5):
    # n is the number of data points.
    # 0 <= p <= 1 is the sampling probability of Male (sex=1).
    keys = ["sex", "age", "work_exp", "hair_length", "hire"]
    data = {key: [] for key in keys}
    for i in range(n):
        rand = np.random.random()
        sex = 1 if rand < p else 0
        age = int(np.random.normal(25, 3))
        hair_length = 35 * np.random.beta(2, 2+5*sex)
        work_exp = int(np.random.poisson(age+6*sex) - np.random.normal(20, 0.2))
        if work_exp < 0:
            work_exp = 0
        hire_rand = np.random.random()
        hire_prob = 1.0/(1+np.exp(25.5-2.5*work_exp))
        # hire_prob = 1.0 / (1 + np.exp(15.5  +10*sex - 2.5 * work_exp))
        # hire_prob = 1.0 / (1 + np.exp(8 - work_exp))
        hire = 1 if hair_length + work_exp > 20 else 0
        data["sex"].append(sex)
        data["age"].append(age)
        data["work_exp"].append(work_exp)
        data["hair_length"].append(hair_length)
        data["hire"].append(hire)
    df = pd.DataFrame(data, columns = keys)
    return df

