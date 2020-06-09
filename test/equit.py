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

np.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    reload(s)
    A = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [1,1,0,1,0,0,0,0],
        [1,0,1,0,0,0,0,0],
        [1,0,0,0,0,1,0,0],
        [1,0,0,0,1,0,1,0],
        [0,0,0,0,0,1,0,0],
        [0,1,1,1,1,1,1,0]
    ])

    def sig(x):
        return anp.tanh(x)
    def sig2(x):
        return -1.*anp.tanh(x)
    def sig3(x):
        return -2.0*anp.tanh(x)
    def zero(x):
        return 0*x
    def f1(x):
        return anp.sin(x)
    def f2(x):
        return 0.9*x + 7/5
    def f3(x):
        return 0.9*x + 0.5

    f = np.array([
        [sig,sig2,sig,sig,sig,sig,sig2,sig],
        [sig2,sig2,sig2,sig2,sig2,sig2,sig2,sig2],
        [sig,sig2,sig,sig,sig,sig,sig2,sig],
        [sig,sig2,sig,sig,sig,sig,sig2,sig],
        [sig,sig2,sig,sig,sig,sig,sig2,sig],
        [sig,sig2,sig,sig,sig,sig,sig2,sig],
        [sig2,sig2,sig2,sig2,sig2,sig2,sig2,sig2],
        [sig,sig2,sig,sig,sig,sig,sig2,sig]
    ])

    a = np.array([zero,f1,f1,f1,f1,f1,f1,f1])
    labels = ['1','2','3','4','5','6','7','8']

    
    counts = []
    for i in range(20):
        G = s.DirectedGraph(A, (a,f), labels=labels)

        for i in range(5):
            base = np.random.choice(G.indices, int(G.n*0.9), replace=False)
            G.specialize(base)

        data = G.coloring()
        for key in data.keys():
            counts.append(len(data[key]))
            
        print(G.n)
    plt.hist(counts)
    plt.show()

    # print(G.coloring())
    # G.network_vis()#use_eqp=True)
    # G.iterate(20,np.random.random(8),graph=True)
    # base = ['1','8','5','6','7']
    # G.specialize(base)
    # G.network_vis(use_eqp=True)
    # with open('half_spec.txt', 'w') as out_file:
    #     out_file.write(str(G.A))
    # G.iterate(20,np.random.random(23),graph=True)
    # G.network_vis(use_eqp=True)
    # print(G.n)
