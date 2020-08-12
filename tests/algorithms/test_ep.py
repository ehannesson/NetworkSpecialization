import numpy as np
import networkx as nx
from scipy import sparse
import os
import time
import multiprocessing

from ep_checker import ep_check

os.chdir('../../core')
from cep import equitablePartition, initialize
os.chdir('../tests/algorithms')


def get_args(color_dict):
    """Returns the `color_inds` and `colors` arguments needed for ep_check()"""
    # sizes of each colorclass
    color_inds = np.array([0] + [len(x) for x in color_dict.values()]).cumsum()
    color_nodes = np.concatenate([x for x in color_dict.values()])

    return color_inds.astype(np.int32), color_nodes.astype(np.int32)


def run_test(G, n_cores):
    """Finds the coarsest equitable partition of G and checks for correctness"""
    if n_cores == -1:
        n_cores = multiprocessing.cpu_count()

    csr_adj = nx.to_scipy_sparse_matrix(G, dtype=np.int32).T
    indptr, adj_inds = csr_adj.indptr, csr_adj.indices

    temp = time.time()
    color_dict = equitablePartition(*initialize(G))
    t1 = time.time() - temp

    temp = time.time()
    res = ep_check(indptr, adj_inds, *get_args(color_dict), n_cores)
    t2 = time.time() - temp

    return bool(res), t1, t2


def random_geometric_check(n, radius, n_cores=-1):
    """Verifies the equitablePartition() function on a Random Geometric Graph"""
    # initialize random graph and adjacency matrix
    G = nx.random_geometric_graph(n, radius)

    return run_test(G, n_cores)


def erdos_renyi_check(n, p, directed=False, n_cores=-1):
    """Verifies the equitablePartition() function on an Erdos-Renyi Graph"""
    G = nx.erdos_renyi_graph(n, p, directed=directed)
    # find and verify coarsest equitable partition
    return run_test(G, n_cores)


def RG_edge_ratio(n, radius, iters=100):
    edges = 0
    for i in range(iters):
        edges += nx.random_geometric_graph(n, radius).number_of_edges()
    return edges/(iters*n)



# def get_args(color_dict):
#     """Returns the `color_inds` and `colors` arguments needed for ep_check()"""
#     # sizes of each colorclass
#     color_inds = np.array([len(x) for x in color_dict.values()]).astype(np.intc)
#     temp = np.zeros_like(color_inds).astype(np.intc)
#     temp0 = np.zeros_like(color_inds).astype(np.intc)
#
#     colors = np.zeros((color_inds.size, color_inds.max()))
#     for key in color_dict.keys():
#         colors[key, :color_inds[key]] = color_dict[key]
#
#     return colors.astype(np.intc), color_inds #, temp, temp0
#
#
# def run_test(G, n_cores):
#     """Finds the coarsest equitable partition of G and checks for correctness"""
#     if n_cores == -1:
#         n_cores = multiprocessing.cpu_count()
#
#     A = nx.to_numpy_array(G).astype(np.intc).T
#     A = np.array(A, order='C')
#
#     temp = time.time()
#     color_dict = equitablePartition(*initialize(G))
#     t1 = time.time() - temp
#
#     temp = time.time()
#     res = ep_check(A, *get_args(color_dict), n_cores)
#     t2 = time.time() - temp
#
#     return bool(res), t1, t2
#
#
# def random_geometric_check(n, radius, n_cores=-1):
#     """Verifies the equitablePartition() function on a Random Geometric Graph"""
#     # initialize random graph and adjacency matrix
#     G = nx.random_geometric_graph(n, radius)
#
#     return run_test(G, n_cores)
#
#
# def erdos_renyi_check(n, p, directed=False, n_cores=-1):
#     """Verifies the equitablePartition() function on an Erdos-Renyi Graph"""
#     G = nx.erdos_renyi_graph(n, p, directed=directed)
#     # find and verify coarsest equitable partition
#     return run_test(G, n_cores)
#
#
# def RG_edge_ratio(n, radius, iters=100):
#     edges = 0
#     for i in range(iters):
#         edges += nx.random_geometric_graph(n, radius).number_of_edges()
#     return edges/(iters*n)
