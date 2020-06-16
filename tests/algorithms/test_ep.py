import numpy as np
import networkx as nx
from scipy import sparse
import os
import time

from ep_checker import ep_check

# os.chdir('../../core')
os.chdir('..')
from ep_test import ep_check as ep_check_bad
os.chdir('../core')
from cep import equitablePartition, initialize
os.chdir('../tests/algorithms')


def get_args(color_dict):
    """Returns the `color_inds` and `colors` arguments needed for ep_check()"""
    # sizes of each colorclass
    color_inds = np.array([len(x) for x in color_dict.values()]).astype(np.int32)

    colors = np.zeros((len(color_inds), np.max(color_inds)), dtype=np.int32)
    for key in color_dict.keys():
        colors[key, :color_inds[key]] = color_dict[key]

    return colors, color_inds


def run_test(G, A):
    """Finds the coarsest equitable partition of G and checks for correctness"""
    temp = time.time()
    color_dict = equitablePartition(*initialize(G))
    t1 = time.time() - temp

    temp = time.time()
    res = ep_check(A, *get_args(color_dict))
    t2 = time.time() - temp

    temp = time.time()
    ep_check_bad(A, color_dict)
    t3 = time.time() - temp

    return res, t1, t2, t3


def random_geometric_check(n, radius):
    """Verifies the equitablePartition() function on a Random Geometric Graph"""
    # initialize random graph and adjacency matrix
    G = nx.random_geometric_graph(n, radius)
    A = nx.to_numpy_array(G).astype(np.int32).T
    # find and verify coarsest equitable partition
    res, t1, t2 = run_test(G, A)

    return run_test(G, A)


def erdos_renyi_check(n, p, directed=False):
    """Verifies the equitablePartition() function on an Erdos-Renyi Graph"""
    G = nx.erdos_renyi_graph(n, p, directed=directed)
    A = nx.to_numpy_array(G).astype(np.int32).T

    # find and verify coarsest equitable partition
    return run_test(G, A)


def RG_edge_ratio(n, radius):
    edges = 0
    for i in range(100):
        edges += nx.random_geometric_graph(n, radius).number_of_edges()
    return edges/(100*n)


def ER_edge_ratio(n, p, directed=False):
    edges = 0
    for i in range(100):
        edges += nx.erdos_renyi_graph(n, p, directed=directed).number_of_edges()
    return edges/(100*n)
