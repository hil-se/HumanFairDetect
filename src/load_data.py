import pandas as pd
from synthetic_data import make_data, make_data2, make_data3

def load(data):
    # data: name of the dataset, can be {"adult", "heart", "compas", "bank", "german"}
    datasets = {"adult": load_adult, "heart": load_heart, "compas": load_compas, "bank": load_bank, "default": load_default, "german": load_german, "student-mat": load_student_mat, "student-por": load_student_por, "synthetic1": load_synthetic1, "synthetic2": load_synthetic2, "synthetic3": load_synthetic3}
    if data not in datasets:
        raise Exception("Unknown dataset name.")
    return datasets[data]()

def load_adult():
    df = pd.read_csv("../data/adult.csv")
    # sensitive attribute names
    A = ["sex", "race"]
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "Male" else 0)
    # discretize race: white vs. non-white
    df['race'] = df['race'].apply(lambda x: 1 if x == "White" else 0)
    # prefer >50K as label 1
    df['income'] = df['income'].apply(lambda x: 1 if x== ">50K" else 0)
    return df, A

def load_heart():
    df = pd.read_csv("../data/heart.csv")
    # sensitive attribute names
    A = ["age"]
    # discretize age: x>60
    df['age'] = df['age'].apply(lambda x: 1 if x > 60 else 0)
    # prefer 0 (< 50% diameter narrowing) as label 1
    df['y'] = df['y'].apply(lambda x: 1 if x==0 else 0)
    return df, A

def load_default():
    df = pd.read_csv("../data/default.csv")
    # sensitive attribute names
    A = ["SEX"]
    df['SEX'] = df['SEX'].apply(lambda x: 0 if x == 2 else 1)
    # prefer 0 (Default Payment = No) as label 1
    df['default payment next month'] = df['default payment next month'].apply(lambda x: 1 if x == 0 else 0)
    return df, A

def load_compas():
    df = pd.read_csv("../data/compas-scores-two-years.csv")
    features_to_keep = ['sex', 'age', 'age_cat', 'race',
                        'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                        'priors_count', 'c_charge_degree', 'c_charge_desc',
                        'two_year_recid']
    df = df[features_to_keep]
    # sensitive attribute names
    A = ["sex", "race"]
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "Male" else 0)
    # discretize race: Caucasian vs. non-Caucasian
    df['race'] = df['race'].apply(lambda x: 1 if x == "Caucasian" else 0)
    # prefer 0 (no recid) as label 1
    df['two_year_recid'] = df['two_year_recid'].apply(lambda x: 1 if x==0 else 0)
    return df, A

def load_bank():
    df = pd.read_csv("../data/bank.csv", sep =";")
    # sensitive attribute names
    A = ["age"]
    # discretize age: x>25
    df["age"] = df["age"].apply(lambda x: 1 if x > 25 else 0)
    # prefer yes as label 1
    df['y'] = df['y'].apply(lambda x: 1 if x=="yes" else 0)
    return df, A

def load_german():
    column_names = ['status', 'month', 'credit_history',
                    'purpose', 'credit_amount', 'savings', 'employment',
                    'investment_as_income_percentage', 'sex',
                    'other_debtors', 'residence_since', 'property', 'age',
                    'installment_plans', 'housing', 'number_of_credits',
                    'skill_level', 'people_liable_for', 'telephone',
                    'foreign_worker', 'credit']
    df = pd.read_csv("../data/german.data", sep=' ', header=None, names=column_names)
    # sensitive attribute names
    A = ["age", "sex"]
    # discretize age: x>25
    df["age"] = df["age"].apply(lambda x: 1 if x > 25 else 0)
    # transform personal_status into sex
    df["sex"] = df["sex"].apply(lambda x: 1 if x in {"A91", "A93", "A94"} else 0)
    # prefer 1 (good credit) as label 1
    df['credit'] = df['credit'].apply(lambda x: 1 if x==1 else 0)
    return df, A

def load_student_mat():
    df = pd.read_csv("../data/student-mat.csv", sep=";")
    # sensitive attribute names
    A = ["sex"]
    # discretize sex
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "M" else 0)
    # prefer yes as label 1
    df['y'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=['G3'])
    return df, A

def load_student_por():
    df = pd.read_csv("../data/student-por.csv", sep=";")
    # sensitive attribute names
    A = ["sex"]
    # discretize sex
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "M" else 0)
    # prefer yes as label 1
    df['y'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=['G3'])
    return df, A

def load_synthetic1():
    df = make_data2(p=0.5, l=0)
    A = ["sex", "age"]
    df["age"] = df["age"].apply(lambda x: 1 if x > 25 else 0)
    return df, A

def load_synthetic2():
    df = make_data2(p=0.5, l=6)
    A = ["sex", "age"]
    df["age"] = df["age"].apply(lambda x: 1 if x > 25 else 0)
    return df, A

def load_synthetic3():
    df = make_data2(p=0.95, l=6)
    A = ["sex", "age"]
    df["age"] = df["age"].apply(lambda x: 1 if x > 25 else 0)
    return df, A
