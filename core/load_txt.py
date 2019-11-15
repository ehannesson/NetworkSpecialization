# read_text.py
from scipy import sparse
import numpy as np
import pickle

def loadtxt(filename, infer_format=True, directed=None, weighted=False, sep=' ',
            save=False):
    """
    Loads a text file into a sparse graph. Assumes a 2 or 3 column format with
    columns separated by sep. The first two columns should specify an edge from
    the first to the second. If directed is False, this will be entered as an
    undirected edge between the first and the second. The third column is optional,
    and should specify the weight of the edge.

    NOTES:
        DOES NOT SUPPORT MULTIGRAPHS. If a multigraph is passed in and weighted
            is true, the duplicates will be summed together into a single connection.
            In the case that weighted is False, duplicate entries are ignored and
            the standard adjacency matrix will be created.
        REMOVES ALL SELF EDGES. Our specialization code cannot handle that yet.

    Parameters:

        filename (str): path to text file

        infer_format (bool): If True (default), attempts to automatically infer
            the format. Even when infer_format=True, other keyword arguments can
            be used to specify certain behaviors (see below).

        directed (bool): If None (default), infer_format must be true; in this
            case, attempts to infer if the graph is directed or not. If True/False,
            graph will be treated as directed/undirected, regardless of if
            infer_format is True of False.

        weighted (bool): If False (default), graph is constructed as unweighted,
            regardless of whether there are weights or not in the file. If True,
            weights are used. Raises a ValueError if True but no weights present.

        sep (str): string to split columns by

        save (bool/str): If False (default), does nothing. If a string specifying
            a save path is given, the csr_matrix is pickled at that location.

    Returns:

        G (sparse.csr_matrix): sparse matrix of graph data
    """
    # dictionaries mapping [un]directed/[un]weighted format strings to bools
    dir_dict = {'sym': False, 'asym': True}
    weight_dict = {'unweighted': False, 'posweighted': True, 'signed': True}

    # load file
    with open(filename, 'r') as f:
        data = f.read()

    data = data.split('\n')
    format_str = data[0].split(sep)[1:] # first line of document is format string

    # sometimes there are other lines with info that isn't needed
    rm_ind = 0
    for elem in range(len(data)):
        # run through the lines of data until we find the first used line
        if data[elem][0] == '%':
            rm_ind = elem + 1
        else:
            break

    data = data[rm_ind:-1]   # last line is empty

    if infer_format:
        # try to infer sep
        if '\t' in data[0]:
            sep = '\t'
        elif ',' in data[0]:
            sep = ','
        # if it isn't this, then it is likely ' ' (default), or user specified

        # split into list of lists; inner lists contain connection data
        data = [entry.strip().split(sep) for entry in data]
        data = np.array(data, dtype=np.int) # numpyify and cast the strings to ints

    if infer_format:
        try:
            # does the document tell us if this is weighted or not?
            weight_format = weight_dict[format_str[1]]
        except KeyError:
            if weighted is False:
                # if the user doesn't care about weights, forget about it
                pass
            else:
                # otherwise, yell at them for doing something confusing
                print(('KeyError: "{}" is an unknown formatting option for '
                        'weighted/unweighted.'.format(format_str[1])))
                print('Current known options are: {}.'.format(weight_dict))
                return None

        # number of columns specifies the formatting
        rows, cols = data.shape

        # weighted inference
        if cols == 2 and weighted:
            # if user specified weighted, but there are no weights
            raise ValueError(f'File "{filename}" has no weights, but weighted was set to True.')
        elif cols == 4 and weighted:
            # if users specified weighted, but we have the weird 4 column format
            if np.all(data[:, 2]==1):
                # if the third column is ones, fourth column is weights
                data = data[:, [0, 1, 3]]   # remove column of ones
            else:
                # if third column not just ones, we don't know what to do
                raise ValueError('Weights specified, but file format unknown')

        elif cols >= 3 and weighted is False:
            # if we have weights, but the user said to ignore them
            data = data[:, :2]

        # directed inference
        if directed is None:
            try:
                directed = dir_dict[format_str[0]]
            except KeyError:
                print(('KeyError: "{}" is an unknown formatting option for '
                        'directed/undirected.'.format(format_str[0])))
                print('Current known options are: {}.'.format(format_str[0]))

    # ------------------------------------------------------------------------ #
    # at this point we should have directed and weighted properly specified

    # explicitly extract row and column indices
    col_inds, row_inds = data[:, 0], data[:, 1]
    # note our convention that G[i, j]=1 means that node[i] recieves from node[j]

    # get proper shape
    shape = max(np.max(row_inds), np.max(col_inds)) + 1
    shape = (shape, shape)
    # construct the sparse graph based on directed/undirected
    if directed:
        if weighted:
            # construct weighted, directed graph
            G = sparse.coo_matrix((data[:, -1], (row_inds, col_inds)),
                                    shape=shape)
                # NOTE THAT THIS SUMS DUPLICATE EDGES
        else:
            # we don't want duplicates to be summed
            unique_edges = {(i, j) for (i, j) in zip(row_inds, col_inds)}
            data = np.array([[i, j] for (i, j) in unique_edges])

            # row_inds/col_inds extraction reverse from above b/c we already
                # accounted for the row <-- col structure
            row_inds, col_inds = data[:, 0], data[:, 1]
            rows = row_inds.size

            # construct unweighted, directed graph
            G = sparse.coo_matrix((np.ones(rows), (row_inds, col_inds)),
                                    shape=shape)
    else:
        # if the graph is undirected, we need to add both i <--> j connections
        # extend row/col inds with col/row inds
        row_inds, col_inds = np.append(row_inds, col_inds), np.append(col_inds, row_inds)
        rows = row_inds.size

        if weighted:
            # extend weights
            weights = np.extend(data[:, 2], data[:, 2])

            # construct weighted, undirected graph
            G = sparse.coo_matrix((weights, (row_inds, col_inds)),
                                    shape=shape)
            # NOTE THAT THIS SUMS DUPLICATE EDGES

        else:
            # remove duplicate edges
            unique_edges = {(i, j) for (i, j) in zip(row_inds, col_inds)}
            data = np.array([[i, j] for (i, j) in unique_edges])

            # row_inds/col_inds extraction reverse from above b/c we already
                # accounted for the row <-- col structure
            row_inds, col_inds = data[:, 0], data[:, 1]
            rows = row_inds.size

            # construct unweighted, undirected graph
            G = sparse.coo_matrix((np.ones(rows), (row_inds, col_inds)),
                                    shape=shape)

    # REMOVE SELF EDGES, WE CANNOT HANDLE THEM YET
    G.setdiag(0)
    G.tocsr()

    if save:
        # pickle the graph
        try:
            with open(save, 'wb') as f:
                pickle.dump(G, f)
        except FileNotFoundError as e:
            # if specified path doesn't exist, print the error, then return G
            print(e)
            return G

    return G
