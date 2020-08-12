# plot_community_dist.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from networkx.generators import random_graphs
import networkx as nx
import sparse_specializer as spec
import progressbar
import sys
import os
from mpi4py import MPI


def community_dist_bar(colors, title=None, logscale=True, barh=False, show=True,
                        save=False, **kwargs):
    """
    Creates a community size distribution bar plot.

    Parameters:

        colors: dict/list of coloring data, or path (str) to pickled data.
            Accepted formats:
                dict:   {comm_num: array(nodes_in_comm)}
                        {comm_size: num_comms}

        title (str): If str is passed, sets the plot title to 'title'

        logscale (bool): Default True. If False, bars are linearly scaled.

        barh (bool): Default False. If True, creates a horizontal bar graph.

        show (bool). Default True. If False, plot is generated but not displayed.
            Useful when generating and saving plots in batch.

        save (bool/str): Default False. If str, should be a path to location to
            save the plot.

        **kwargs: other keyword arguments to pass to plt.bar

    Returns: None
    """
    # if show is False, use gui-less backend
    if not show:
        plt.switch_backend('agg')

    # if colors is path data, load it
    if type(colors) is str:
        # load file
        with open(colors, 'rb') as f:
            colors = pickle.load(f)

    # type of dict values should be int or array (or tuple)
    dict_type = type(list(colors.values())[0])
    if dict_type is int:
        # if dict_type is int, it maps comm_size --> num_comms
        # comm_size is bar positions, num_comms is bar heights
        comm_size, num_comms = colors.keys(), colors.values()
    else:
        # if dict_type is array or tuple, we have to extract comm_sizes
        # get largest community size (+1 for range indexing)
        max_comm_size = max([len(comm) for comm in list(colors.values())])+1
        # convert dict to {comm_size: num_comms}
        colors = {i: sum([1 for k in list(colors.keys()) if len(colors[k])==i])
                    for i in range(1, max_comm_size)}
        # remove comm_sizes with num_comms=0 to reduce plotting spatial complexity
        colors = {size: num for (size, num) in colors.items() if num!=0}
        # comm_size is bar positions, num_comms is bar heights
        comm_size, num_comms = colors.keys(), colors.values()

    # create plot
    if barh:
        raise NotImplementedError('barh plots not implemented')
    else:
        plt.bar(comm_size, num_comms, log=logscale, **kwargs)

    if title:
        plt.title(title)

    # save plot if specified
    if save:
        plt.savefig(save)

    # display spot
    if show:
        plt.show()


def community_dist_hist(colors, title=None, logscale=True, show=True, save=False,
                        **kwargs):
    """
    Creates a community size distribution histogram.

    Parameters:

        colors: dict of coloring data, or path (str) to pickled data.
            Accepted formats:
                dict:   {comm_num: array(nodes_in_comm)}
                        {comm_size: num_comms}

        title (str): If str is passed, sets the plot title to 'title'

        logscale (bool): Default True. If False, bars are linearly scaled.

        show (bool). Default True. If False, plot is generated but not displayed.
            Useful when generating and saving plots in batch.

        save (bool/str): Default False. If str, should be a path to location to
            save the plot.

        **kwargs: other keyword arguments to pass to plt.hist:
            bins (int, seq): if int, bins+1 bin edges are used
                             if seq, gives the bin edges
            orientation (str): 'horizontal', 'vertical'

    Returns: None
    """
    # if show is False, use gui-less backend
    if not show:
        plt.switch_backend('agg')

    # if colors is path data, load it
    if type(colors) is str:
        # load file
        with open(colors, 'rb') as f:
            colors = pickle.load(f)

    # type of dict values should be int or array (or tuple)
    dict_type = type(list(colors.values())[0])
    if dict_type is int:
        # if dict_type is int, it maps comm_size --> num_comms
        comm_sizes = []
        for size in colors.keys():
            # community sizes list; entry for each community of size comm
            comm_sizes += [size]*colors[size]
    else:
        # if dict_type is array or tuple, extract comm_sizes entry by entry
        comm_sizes = [len(comm) for comm in colors.values()]

    # create plot
    plt.hist(np.log(comm_sizes), log=logscale, **kwargs)
    plt.xlabel('Community Size (num nodes)')
    plt.ylabel('Number of Communites')

    if title:
        plt.title(title)

    if save:
        plt.savefig(save)

    if show:
        plt.show()


def erdos_renyi_colorings(n=100, p=0.1, graphs=10, edges=None, agg=True,
                          verbose=True, MPI=False):
    """
    Generates erdos-renyi random graphs, computes the coarsest equitable
    coloring of each, and returns the coloring statitstics.

    Parameters:
        n (int): number of nodes to be used in

        p (float): probability that any two given nodes are connected; this uses
            the G(n, p) erdos-renyi scheme. If edges is specified, G(n, m) is
            used instead and this parameter is ignored.

        graphs (int): number of erdos-renyi graphs to generate and run coloring
            statistics on

        edges (int): if specified, graphs are drawn uniformly from the space of
            all possible graphs with n nodes and m edges. In this case, the
            parameter p is ignored.

        agg (bool): If True (default), the coloring statistics are aggregated
            together and a single dictionary is returned. Otherwise, a list of
            dictionaries is returned, each of which contains the coloring
            statistics for one graph.

        verbose (bool): If True (default), a progress bar will be displayed.
            Otherwise, you'll be left in the dark.

        MPI (int): If int is specified, uses MPI to run in parallel with MPI
            processes

    Returns:
        color_stats: If agg=True, the statistics for each coloring are aggregated
            and a single average statistic is returned. Otherwise, a list of the
            coloring statistics is returned.
    """
    if MPI:
        # run the secret version via command-line
        os.system(f'mpiexec -n {MPI} python statistics.py erc {n} {p} {graphs} '
                  f'{edges} {agg} {verbose}')

        # load pickled files
        colorings = []
        for _ in range(1, MPI+1):
            with open(f'temp_{_}', 'rb') as f:
                colorings += pickle.load(f)
            # remove temp file
            os.system(f'rm temp_{_}')

        if agg:
            # aggregate coloring information
            agg_colors = dict()
            for graph in colorings:
                comm_sizes = [len(comm) for comm in graph.values()]
                for size in comm_sizes:
                    try:
                        agg_colors[size] += 1
                    except KeyError:
                        agg_colors[size] = 1

            colorings = agg_colors

        return colorings

    else:
        return _erdos_renyi_colorings(n=n, p=p, graphs=graphs, edges=edges,
                                        agg=agg, verbose=verbose)


def _erdos_renyi_colorings(n=100, p=0.1, graphs=10, edges=None, agg=True,
                          verbose=True, MPI=False):
    """
    Secret backend that actually does the work for erdos_renyi_colorings. This
    is necessary to allow parallelism with MPI.

    For documentation, see the docstring for erdos_renyi_colorings.
    """
    # set verbosity
    if verbose:
        graphs = progressbar.progressbar(range(graphs))
    else:
        graphs = range(graphs)

    colorings = []  # holds each graph's coloring stats

    for g in graphs:
        # generate graph
        if edges is None:
            if p < 0.2:
                # if probability is low, this is faster than the other one
                G = nx.random_graphs.fast_gnp_random_graph(n, p, directed=True)
            else:
                G = nx.random_graphs.gnp_random_graph(n, p, directed=True)
        else:
            if edges/(n*(n-1)) < 0.25:
                # this is faster with sparse graphs
                G = nx.random_graphs.gnm_random_graph(n, edges, directed=True)
            else:
                G = nx.random_graphs.dense_gnm_random_graph(n, edges, directed=True)

        # create specializer graph object
        G = spec.DirectedGraph(nx.adjacency_matrix(G))
        # run coloring (note that nx transposes from our convention, but the
        # community structure will be the same)
        G.coloring()
        colorings.append(G.colors)

    if agg:
        # aggregate coloring data
        agg_colors = dict()
        for graph in colorings:
            comm_sizes = [len(comm) for comm in graph.values()]
            for size in comm_sizes:
                try:
                    agg_colors[size] += 1
                except KeyError:
                    agg_colors[size] = 1

        colorings = agg_colors

    if MPI:
        with open(f'temp_{MPI}', 'wb') as outfile:
            pickle.dump(colorings, outfile)

    return colorings


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        pass
    elif args[1] == 'erc':
        # runninc erdos_renyi_colorings in parallel
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()+1
        size = comm.Get_size()

        n, p, graphs, edges, agg, verbose = args[2:]
        n, p, graphs = int(n), float(p), int(graphs)
        if edges == 'None':
            edges = None
        else:
            edges = int(edges)

        if agg == 'True':
            agg = True
        else:
            agg = False

        if verbose == 'True':
            verbose = True
        else:
            verbose = False

        # distribute graphs among nodes
        graphs = graphs//size
        colorings = _erdos_renyi_colorings(n=n, p=p, graphs=graphs, edges=edges,
                                            agg=False, verbose=verbose, MPI=rank)
