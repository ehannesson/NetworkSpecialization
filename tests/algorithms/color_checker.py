import numpy as np
import networkx as nx
import scipy.sparse as sparse
import os
import time
import pickle
# from joblib import Parallel, delayed
from multiprocessing import Pool
import matplotlib.pyplot as plt

os.chdir('../core')
import sparse_specializer as spec
import coarsest_equitable_partition as cep
os.chdir('../unit_tests')

def color_checker(A, colors, n_jobs=8):
    """
    Function that can make sure our coloring is equitable.
    """
    _colors = list(colors.values())
    # chunksize = len(_colors)//n_jobs + 1
    # p = Pool(n_jobs)

    # def _check_colors(check, all_colors=_colors, p=p, A=A):
        # """Helper function for parallel computation"""
    for color1 in _colors:
        for color2 in _colors:
            x = np.sum(A[color1][:, color2], axis=1)
            if not (x==x[0]).all():
                # p.close()
                return False
    return True

    # res = np.all(p.imap(_check_colors, _colors, chunksize=chunksize))
    # print(res)
    # p.terminate()

    # return res


def random_test(n, radius, kind=None, print_=True, return_time=False, return_graph=False):
    G = nx.random_geometric_graph(n, radius)
    A = nx.to_scipy_sparse_matrix(G)
    G = spec.DirectedGraph(A, None)

    start_time = time.time()
    ep = cep.coarsest_equitable_partition(G)
    ttime = time.time() - start_time
    if print_:
        print(f'CEP Time:\t{ttime}')

    res = color_checker(A, ep)

    if print_:
        if res:
            print('SUCCESS!')
        else:
            print('FAIL -_-')

    if return_time and return_graph:
        return res, ttime, G
    elif return_time:
        return res, ttime
    elif return_graph:
        return res, G
    else:
        return res


def r_check(n, radius, kind=None):
    G = nx.random_geometric_graph(n, radius)
    A = nx.to_scipy_sparse_matrix(G)
    G = spec.DirectedGraph(A, None)
    print(f'Nodes:\t{n}\nEdges:\t{G.A.nnz}')

    start_time = time.time()
    ep = cep.coarsest_equitable_partition(G, kind=kind)
    print(f'CEP:\t{time.time()-start_time}')

    start_time = time.time()
    try:
        G.coloring()
    except KeyboardInterrupt:
        print(f'Interrupted. Time Elapsed:\t{time.time() - start_time}')
        if color_checker(A, ep):
            print('SUCCESS!')
        return ep, None

    print(f'SPEC:\t{time.time()-start_time}')
    res = color_checker(A, ep)
    if res:
        print('SUCCESS!')
    else:
        print('FAIL -_-')

    return G


def findFailures(n=7, radius=0.5, iters=50, kind=None, verbose=False,
                 remove_old=True, base_path='incorrect_colorings/', debug=True):
    """
    Tests the coarsest_equitable_partition() function by coloring `iters` random
    geometric graphs (with `n` points and connection distance `radius`), tracking
    the number of incorrect colorings. Saves the sparse adjacency matrix for each
    incorrect coloring.
    """
    # find filename of last incorrect coloring saved
    files = os.listdir(base_path)
    if remove_old:
        if len(files) != 0:
            os.system(f'rm -r {base_path}/*')
        fnum = 1
    else:
        if len(files) == 0:
            fnum = 1
        else:
            last_file = sorted(files)[-1][:-4]
            fnum = int(last_file) + 1

    fname = ('000' + str(fnum))[-3:]

    failures = 0
    ep_time = 0
    check_time = 0

    for _ in range(iters):
        # get sparse adjacency of random geometric graph
        A = nx.to_scipy_sparse_matrix(nx.random_geometric_graph(n, radius))
        # create DirectedGraph object
        G = spec.DirectedGraph(A, None)

        # find equitable partition
        start_time = time.time()
        debug_ep, debug_new_colors, debug_ssets = cep.coarsest_equitable_partition(G, kind=kind, debug=True)
        ep_time += time.time() - start_time

        # check if it is equitable
        start_time = time.time()
        if debug:
            check = color_checker(A, debug_ep[-1])
        else:
            check = color_checker(A, debug_ep)
        check_time += time.time() - start_time

        if not check:
            # we failed :(
            failures += 1
            print('FAILED')
            # save adjacency matrix of the failure
            sparse.save_npz(base_path + fname + '_A.npz', A)

            exts = ['_ep', '_new_colors', '_ssets']
            debug_data = [debug_ep, debug_new_colors, debug_ssets]
            for _ in range(3):
                with open(base_path + fname + exts[_], 'wb') as f:
                    # save the failed coloring
                    pickle.dump(debug_data[_], f)

            # increment file name
            fnum += 1
            fname = ('000' + str(fnum))[-3:]

        if verbose and not _%100:
            print(f'Iteration:\t{_}')

    return failures, ep_time/iters, check_time/iters


def checkFailures(base_path='incorrect_colorings/', debug=True):
    # get paths to failures
    failures = os.listdir(base_path)
    failures.sort()
    adj = failures[::4]         # adjacency files
    ep = failures[1::4]         # coloring files
    n_colors = failures[2::4]   # new color files
    structs = failures[3::4]      # structure set files

    files = [adj, ep, n_colors, structs]

    for (A, coloring, ncs, ssets) in zip(adj, ep, n_colors, structs):
        # load adjacency and partition objects
        A = sparse.load_npz(base_path + A)

        exts = [coloring, ncs, ssets]
        data = []
        for ext in exts:
            with open(base_path + ext, 'rb') as f:
                data.append(pickle.load(f))

        if debug:
            # then coloring is the sequence of colorings
            # find number of rows needed for plot, assuming 3 columns per row
            rows = int(np.ceil((len(data[0])+1)/3))
            fig, ax = plt.subplots(rows, 3)
            ax = np.ravel(ax)


            # create DirectedGraph object
            G = spec.DirectedGraph(A, None)

            nc, ss = data[1:]
            for _ in range(len(data[0])):
                # set G's coloring to be the `plot` iteration of the incorrect coloring
                G.colors = data[0][_]
                # display incorrect partitioning of G
                G.network_vis(ax=ax[_], use_eqp=True, circular_layout=True)
                # construct string for the title
                title = 'New Colors: ' + str(nc[_])
                # _temp = [str(key) + ': ' + str(val) for (key, val) in ss.items()]
                if ss[_] != '':
                    _print = '\nIteration ' + str(_) + ' Structure Sets:'
                    _print += '\n\tColor * Structure Sets:\n'
                    _print += '\t(vertex, num neighbors of color *)\n'
                    for ss_color in ss[_]:
                        _print += f'\t\t{ss_color}\n'

                    # print(ss)
                    # print(ss[_])
                    # if _ > 2:
                        # raise ValueError
                    # _colors = [list(_sset.keys())[0] for _sset in ss[_]] # list of structure set colors (int)
                    # _nodes = [ss[_][x] for x in range(len(ss[_]))]
                    #
                    # for _color in range(len(_colors)):
                    #     _print = '\nIteration ' + str(_) + ' Structure Sets:'
                    #     _print += '\n\tColor * Structure Sets:'
                    #     for _node in ss[_]:
                    #         _node_keys = list(_node.keys())
                    #         _temp = [str(_node[_key][0]) for _key in _node_keys]
                    #         _temp_str = ''
                    #         for _entry in _temp:
                    #             _temp_str += '\n\t\t' + _entry
                    #
                    #         _print += _temp_str
                    #
                    #     _print += '\n'

                    print(_print)



                    # _outer_keys = [list(x.keys())[0] for x in _sets] # list of integer keys
                    # _tuples = [_sets[x][_outer_keys[x]] for x in range(len(_sets))] # list of tuples for each color
                    # _inner_keys = [[list(x.keys())[0] for x in _tup] for _tup in _tuples]
                    # _inner_vals = '\n'.join([str(x) for x in _inner_keys])
                    #
                    # _pairs = [str(_outer_keys[x]) + ': ' + str(_inner_keys[x][y]) + '\n\t' + str(_inner_vals)
                    #             for x in range(len(_outer_keys)) for y in range(len(_inner_keys))]
                    #             # for _out_key, _inner_key, _pair in zip(_outer_keys, _inner_keys, _tuples)]
                    # print('\n\n'.join([str(_pair) for _pair in _pairs]))
                    # _color_tuple_pairs = [[(key, _sets[key]) for key in _color.keys()] for _color in _sets]
                    # print(_color_tuple_pairs)
                    # _temp = '\nStructure Sets\n' + '\n'.join([str(x) for x in _sets])
                    # print(_temp)
                    # title += _temp

                ax[_].set_title(title)

            # find the correct partitioning of G
            G.coloring()

            # display it on the axis after the final incorrect iteration
            G.network_vis(ax=ax[-1], use_eqp=True,
                          title='Correct Partition', circular_layout=True)

            # pause until ready to continue
            user_in = input("Press Enter to Continue")

        else:
            # create DirectedGraph object
            G = spec.DirectedGraph(A, None)
            # create figure for plotting partitions
            fig, ax = plt.subplots(1, 2)

            # find the correct partitioning of G
            G.coloring()
            # display it on the first axis
            G.network_vis(ax=ax[0], use_eqp=True, title='Correct Partition', circular_layout=True)

            # set G's coloring to be incorrect partition
            G.colors = data[0]
            # display incorrect partitioning of G
            G.network_vis(ax=ax[1], use_eqp=True, title='Incorrect Partition', circular_layout=True)

            # pause until ready to continue
            user_in = input("Press Enter to Continue")
