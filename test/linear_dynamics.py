import sys
import os
path = os.getcwd()
sys.path.insert(1, path[:-4])

import core.specializer as s
import numpy as np
from importlib import reload
import networkx as nx
import matplotlib.pyplot as plt
import time
import autograd.numpy as anp

if __name__ == "__main__":

    def doubling(x):
        return 2*x % 1

    def lin1(x):
        return .1*x
    
    def lin2(x):
        return (1/3) * x

    def sig1(x):
        return anp.tanh(x)
    
    def iden(x):
        return x
    
    def zero(x):
        return 0*x

    reload(s)
    A = np.array([
        [0,0,1,1],
        [1,0,1,1],
        [0,1,0,1],
        [0,1,1,0]
    ])

    f = np.array([
        [zero,zero,lin2,lin1],
        [lin2,zero,lin1,lin1],
        [zero,lin1,zero,lin1],
        [zero,lin1,lin1,zero]
    ])

    # a = np.array([sig1,sig1,sig1,sig1])
    a = np.array([sig1,zero,zero,zero])
    labels = ['1','2','3','4']

    G = s.DirectedGraph(A, (a,f), labels=labels)
    # G.coloring()
    # print(G.colors)

    # G.iterate(200, np.random.random(G.n) * 15, graph=True)

    base = ['1']
    G.specialize(base)
    G.coloring()
    print(G.colors)
    G.iterate(200, np.random.random(G.n), graph=True)
    # base = np.random.choice(G.indices, int(G.n*0.9), replace=False)
    # G.specialize(base)

    # G.iterate(200, np.random.random(G.n) * 15, graph=True)

