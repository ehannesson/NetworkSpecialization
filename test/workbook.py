# import sys
# import os
# path = os.getcwd()
# sys.path.insert(1, path[:-4])

import core.specializer as s
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
    f = np.array([
        [zero,sig,sig,zero,zero,zero],
        [sig,zero,sig,sig,zero,zero],
        [sig,sig,zero,sig,zero,zero],
        [zero,sig,sig,zero,sig,sig],
        [zero,zero,zero,sig,zero,sig],
        [zero,zero,zero,sig,sig,zero]
    ])
    a = np.array([sig, logistic , logistic, sig, sig, sig])
    labels = ['1', '2', '3', '4','5','6']

    G = s.DirectedGraph(B, (a,f), labels=labels)
    x0 = np.random.random(G.n)
    x0[1] = x0[2] + 1e-10
    # G.iterate(100, x0, graph=True)

    base = ['1','4', '5','6']
    G.specialize(base)
    # G.network_vis()
    x0 = np.random.random(G.n)
    x0[1] = x0[2] + 1e-10
    G.iterate(100, np.random.random(G.n), graph=True)
    # print(G.n)
    # G.iterate_with_perturbations(300, np.random.random(G.n), ([150], 10), graph=True, save_img=True, title='../graphs/chaotic cgnn with perturbation')