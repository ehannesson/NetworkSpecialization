# grow.py

import numpy as np
import networkx as nx
import networkx.algorithms as nxalg


import subgraph_specializer as ns


class Graph(ns.Graph):
    """
    Extends the ns.Graph class with rules for selecting nodes for specialization.
    """

    def __init__(self, A, labels=None, F=None, directed=False):
        """
        A sparse graph object with methods designed to facilitate
            network analysis

        Attributes
        ----------
            A (scipy.sparse.csr_matrix)(n,n): the sparse adjacency matrix
                of the network. A[i,j] denotes the weight on the edge from
                node j to node i.
            n (int): number of nodes
            labels (list(str)): list of labels assigned to the nodes of the
                network
            labeler (dict(int, str)): maps indices to labels
            indexer (dict(str, int)): maps labels to indices


        Methods
        -------
            specialize()
            coloring()
        """
        super().__init__(A, labels, F)
        self.directed = directed

        if self.directed:
            self.nxG = nx.DiGraph(A.T)
        else:
            self.nxG = nx.Graph(A)

    def _select_specializing_set(self, p):
        """
        Helper function to select nodes for the specializing set.

        Parameters
        ----------
        p : float (0<`p`<1)
            Percent of nodes to include in the specializing set.

        Returns
        -------
        spec_set : ndarray(int)
            Sorted ndarray with indices of nodes in the specializing set.
        """

        size = round(self.n*p)
        # uniformly sample nodes (w/o replacement)
        spec_set = np.random.default_rng().choice(self.indices, size=size, replace=False)
        # sort inplace
        spec_set.sort()

        return spec_set

    def _get_strongly_connected_specializing_set(self, spec_set):
        """
        Finds the largest strongly connected component in the specializing set.

        This only applies to directed graphs, for undirected graphs, use
        _get_connected_specializing_set() instead.

        Parameters
        ----------
        spec_set : ndarray(int)
            Sorted ndarray with indices of nodes in the specializing set.

        Returns
        -------
        set
            Set of labels of nodes in the largest strongly connected component of
            the specializing subgraph (subset).

        Raises
        ------
        NetworkXNotImplemented
            If self.nxG is undirected.
        """

        specializing_subgraph = self.nxG.subgraph(spec_set).copy()
        return max(nx.strongly_connected_components(sub_G), key=len)

    def _get_weakly_connected_specializing_set(self, spec_set):
        """
        Finds the largest weakly connected component in the specializing set.

        This only applies to directed graphs, for undirected graphs, use
        _get_connected_specializing_set() instead.

        Parameters
        ----------
        spec_set : ndarray(int)
            Sorted ndarray with indices of nodes in the specializing set.

        Returns
        -------
        set
            Set of labels of nodes in the largest weakly connected component of
            the specializing subgraph (subset).

        Raises
        ------
        NetworkXNotImplemented
            If self.nxG is undirected.
        """

        specializing_subgraph = self.nxG.subgraph(spec_set).copy()
        return max(nx.weakly_connected_components(sub_G), key=len)

    def _get_connected_specializing_set(self, spec_set):
        """
        Finds the largest connected component in the specializing set.

        Parameters
        ----------
        spec_set : ndarray(int)
            Sorted ndarray with indices of nodes in the specializing set.

        Returns
        -------
        set
            Set of labels of nodes in the largest connected component of the
            specializing subgraph (subset).

        Raises
        ------
        NetworkXNotImplemented
            If self.nxG is directed.
        """

        specializing_subgraph = self.nxG.subgraph(spec_set).copy()
        return max(nx.connected_components(sub_G), key=len)

    def _specialize_grow(self, component_type, p=0.98):
        """
        Performs one step in the specialization-growth process.

        Parameters
        ----------
        component_type : str {'connected', 'weak', 'strong'}
            Specifies what type of component to use when selecting the specializing
            set. 'Connected' is valid only for undirected graphs, while 'strong'
            and 'weak' are valid only for directed graphs.

        p : float (0<`p`<1)
            Percent of nodes to include in the specializing set.
        """

        spec_set = self._select_specializing_set(p)

        if component_type == 'connected':
            connected_spec_set = self._get_connected_specializing_set(spec_set)
        elif component_type == 'weak':
            connected_spec_set = self._get_weakly_connected_specializing_set(spec_set)
        elif component_type == 'strong':
            connected_spec_set = self._get_strongly_connected_specializing_set(spec_set)
        else:
            raise ValueError("`component_type` must be one of {'connected', 'weak', 'strong'}")
