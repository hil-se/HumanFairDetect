import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import make_data
from collections import Counter
from demos import cmd

def load_synthetic2():
    return make_data(n=10000, p=0.5), ["sex"]

def figB():
    f = lambda x : 1.0/(1+np.exp(25.5-2.5*x))
    xs = range(0,30,1)
    y1s = np.array([f(x) for x in xs])
    y2s = 1.0 - y1s
    plt.plot(xs, y1s)
    plt.plot(xs, y2s)
    plt.savefig("../figs/figB.png")

def figS(sex = 1):
    df, A = load_synthetic2()
    X = df[df["sex"]==sex]
    XP = X[df["hire"]==1]
    XN = X[df["hire"] == 0]
    YP = Counter(XP["work_exp"])
    YN = Counter(XN["work_exp"])
    xs = range(0, 30, 1)
    yps = [YP[x] for x in xs]
    yns = [YN[x] for x in xs]
    plt.plot(xs, yps)
    plt.plot(xs, yns)
    plt.savefig("../figs/figS"+str(sex)+".png")

def figW(sex = 1):
    df, A = load_synthetic2()
    X = df[df["sex"]==sex]
    XP = X[df["hire"]==1]
    XN = X[df["hire"] == 0]
    YP = Counter(XP["work_exp"])
    YN = Counter(XN["work_exp"])
    NP = sum(YP.values())
    NN = sum(YN.values())
    xs = range(0, 30, 1)
    yps = [YP[x]/NP for x in xs]
    yns = [YN[x]/NN for x in xs]
    plt.plot(xs, yps)
    plt.plot(xs, yns)
    plt.savefig("../figs/figW"+str(sex)+".png")

if __name__ == "__main__":
    eval(cmd())