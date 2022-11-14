### Testing Fairness in Human Decisions With Machine Learning Algorithmic Bias

#### Data (included in the [data/](https://github.com/hil-se/HumanFairDetect/tree/master/data) folder)

 - Adult, Bank, and Heart datasets
   + Raw data comes from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).
 - Compas dataset
   + Raw data comes from [propublica](https://github.com/propublica/compas-analysis/)

#### Function

  - inject_exp_all() function in the [src/main.py](https://github.com/hil-se/HumanFairDetect/blob/master/src/main.py#L126) file runs all experiments with injected bias.
  - one_exp() function in the [src/exp.py](https://github.com/hil-se/HumanFairDetect/blob/master/src/exp.py#L28) file has the main pipeline for each experiment.
  - inject_bias() function in the [src/exp.py](https://github.com/hil-se/HumanFairDetect/blob/master/src/exp.py#L111) file injects bias to the training data.
  - FairBalance() function in the [src/preprocessor.py](https://github.com/hil-se/HumanFairDetect/blob/master/src/preprocessor.py#L59) file balances the class distribution within each demographic group.
  - Metrics class in the [src/metrics.py](https://github.com/hil-se/HumanFairDetect/blob/master/src/metrics.py) file calculates the metrics of accuracy, f1 score, EOD, and AOD.


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

