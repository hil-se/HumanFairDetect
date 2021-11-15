### A Pilot Study on Detecting Unfairness in Human Decisions With Machine Learning Algorithmic Bias Detection

#### Data

 - Adult, Bank, and Heart datasets
   + Raw data comes from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php).
   + Pre-processed by [AIF360](https://github.com/Trusted-AI/AIF360).
 - Compas dataset
   + Raw data comes from [propublica](https://github.com/propublica/compas-analysis/)
   + Pre-processed by [AIF360](https://github.com/Trusted-AI/AIF360).

#### Usage
0. Install dependencies:
```
pip install -r requirements.txt
```
1. Navigate to the source code:
```
cd src
```
2. Generate results in _results/_
```
python main.py
```

#### Experiment on the UTKFace dataset

https://anonymous.4open.science/r/image_fairness-56CC

#### Acknowledgement
This work is built on [AIF360](https://github.com/Trusted-AI/AIF360). The [aif360](https://github.com/hil-se/FairBalance/tree/main/aif360) folder is directed cloned from the [AIF360](https://github.com/Trusted-AI/AIF360) repo on May 1st 2021. It is a great platform facilitating the creation and reproduction of AI bias mitigation algorithms.
