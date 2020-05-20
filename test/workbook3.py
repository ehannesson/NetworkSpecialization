import core.subgraph_specializer as s
import numpy as np
from importlib import reload
import networkx as nx
import matplotlib.pyplot as plt
import time
import autograd.numpy as anp


if __name__ == "__main__":
    reload(s)

    def sig(x):
        return anp.tanh(x)
    def sig2(x):
        y = -2.0*anp.tanh(x)
        return y
    def zero(x):
        return 0*x
    def logistic(x):
        x = x % 1
        return 4*x*(1-x)
    def doubling_map(x):
        if x > .5:
            return 2*x
        else:
            return 2*x - 2
    def func1(x):
        y = (9/10)*x + 1.75
        return y
    def func2(x):
        y = (9/10)*x + 1.25
        return y
    def func3(x):
        y = (9/10)*x + 0.5
        return y

    # basic cohen-grossberg neural network
    B = np.array([
        [0,1,1,0,0,0],
        [1,0,1,1,0,0],
        [1,1,0,1,0,0],
        [0,1,1,0,1,1],
        [0,0,0,1,0,1],
        [0,0,0,1,1,0]
    ])
    F = np.array([
        [sig,sig,sig,zero,zero,zero],
        [sig,logistic,sig,sig,zero,zero],
        [sig,sig,logistic,sig,zero,zero],
        [zero,sig,sig,sig,sig,sig],
        [zero,zero,zero,sig,sig,sig],
        [zero,zero,zero,sig,sig,sig]
    ])
    labels = ['1', '2', '3', '4','5','6']

    G = s.Graph(B,labels,F)
    x0 = np.random.random(G.n)
    G.iterate(100, x0, graph=True)