# test_coloring.py
import sys
sys.path.insert(0,'/home/adam/Documents/Research/NetworkSpecialization/core/')
import specializer as spec
import statistics as stats
import numpy as np
import random
from importlib import reload

if __name__ == "__main__":
    reload(spec)
    def zero(x):
        return 0

def random_matrix(n):
    A = np.random.randint(0,2,(n,n))
    A *= (np.eye(n) == 0)
    f = [[zero for _ in range(n)] for _ in range(n)]
    f = np.array(f)
    a = np.diag(f)

    return (A, f, a)

def test(n,iters):
    A,f,a = random_matrix(n)
    labels = [str(i) for i in range(1,n+1)]
    G = spec.DirectedGraph(A,(a,f),labels=labels)
    G.coloring()
    print(G.colors)
    stats.community_dist_hist(G.colors,title=G.n, logscale=False, show=True, save=False)

    for k in range(iters):
        base = random.sample(G.indexer.keys(),int(np.floor(G.n*.9)))
        print('\nSpecializing on ',base)
        G.specialize(base,recolor=True)
        print(G.colors)
        stats.community_dist_hist(G.colors,title='$S_{}$'.format(k+1) + str(base) + 'n=' + str(G.n),logscale=False,show=True,save=False)
