### Testing Fairness in Human Decisions With Machine Learning Algorithmic Bias

#### Data (included in the [data/](https://github.com/hil-se/HumanFairDetect/tree/master/data) folder)

 - Adult, Bank, and Heart datasets
   + Raw data comes from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).
 - Compas dataset
   + Raw data comes from [propublica](https://github.com/propublica/compas-analysis/)

#### Usage
0. Install dependencies:
```
pip install -r requirements.txt
```
1. Navigate to the source code:
```
cd src
```
2. Generate results in _inject\_results/_
```
python main.py inject_exp_all
```

