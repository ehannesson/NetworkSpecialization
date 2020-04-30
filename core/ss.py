import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from copy import deepcopy
import time



class Graph:
    '''
    A sparse graph object that with methods designed to facilitate
        network analysis

    Attributes:
        A (scipy.sparse.csr_matrix)(n,n): the sparse adjacency matrix
            of the network. A[i,j] denotes the weight on the edge from
            node j to node i.
        n (int): number of nodes
        labels (list(str)): list of labels assigned to the nodes of the
            network
        labeler (dict(int, str)): maps indices to labels
        indexer (dict(str, int)): maps labels to indices


    Methods:
        specialize()
        coloring()

    '''

    def __init__(self, A, labels=None):
        '''
        Parameters:
            A (ndarray)(n,n): The adjecency matrix to a directed graph
                where A[i,j] is the weight of the edge from node j to
                node i
            Labels (list(str)): labels for the nodes of the graph,
                defaults to 0 indexing
        '''
        
        n,m = A.shape
        if n != m :
            raise ValueError('Matrix not square')
        # if np.diag(A).sum() != 0:
        #     raise ValueError('Some nodes have self edges')

        # Determine how the labels will be created
        # defaults to simple 0 indexed labeling
        if type(labels) == type(None):
            labels = [f'{i}' for i in range(n)]
        elif (
            type(labels) != type(list()) or
            len(labels) != n or
            type(labels[0]) != type(str())
            ):
            raise ValueError('labels must be a string list of length n')

        # save A as a csr_matrix
        if sp.isspmatrix_csr(A):
            self.A = A
        else:
            self.A = sp.csr_matrix(A)
        self.n = n
        self.labels = labels
        self.indices = np.arange(n)
        self.labeler = dict(zip(np.arange(n), labels))
        self.indexer = dict(zip(labels, np.arange(n)))
        self.origin = self.indexer.copy()
    

    def _update_indexer(self):
        '''
        This function assumes that self.labeler is correct in its
        labeling and returns a new indexer based on that
        '''

        indices = self.labeler.keys()
        labels = self.labeler.values()
        return dict(zip(labels, indices))


    def _update_labeler(self):
        '''
        This function assumers that self.indexer is correct in its
        indexing and returns a new labeler based on that
        '''

        labels = self.indexer.keys()
        indices = self.indexer.values()
        return dict(zip(indices, labels))
        

    def specialize(self, base):
        '''
        Specialize a network around a base set according to the
        specialization model described in:
            Spectral and Dynamic Consequences of Network Specialization
            https://arxiv.org/abs/1908.04435
        
        Parameters:
            base (list(int or str)): list of base nodes by either their
                index or labels. Other nodes become the specialized set
            verbose (bool): print key information
        '''

        # if the base was given as a list of nodes then we convert them 
        # to the proper indices
        if type(base[0]) == str:
            for i, k in enumerate(base):
                base[i] = self.indexer[k]
        elif type(base[0]) != int:
            if not np.issubdtype(base[0], np.integer):
                raise ValueError('base set must be either a list of labels or a'
                'list of indices')
        if type(base) != list:
            base = list(base)
        if len(base) > self.n:
            raise ValueError('base list is too long')

        # permute A so that it is block diagonal with the base first
        spec_set = [i for i in self.indices if i not in base]
        permute = base + spec_set

        # Change the labeler and update the indexer
        self.labeler = {i : self.labeler[permute[i]] for i in range(self.n)}
        self.indexer = self._update_indexer()
        self.labels = [self.labels[i] for i in permute]


        # rearrange the indices to put the base set first
        self.A = self.A[permute,:][:,permute]

        # count the number of way into and out of the specialization set
        base_len = len(base)
        spec_len = len(spec_set)
        in_graph = self.A[base_len:,:][:,:base_len]
        out_graph = self.A[:base_len,:][:,base_len:]
        num_in = in_graph.sum()
        num_out = out_graph.sum()
        num_copies = int(num_in * num_out)
        in_edges = np.argwhere(in_graph != 0)
        out_edges = np.argwhere(out_graph != 0)


        # create the new specialized matrix as a sparse diagonal matrix
        S = sp.block_diag(
            [self.A[:base_len,:][:,:base_len]] + 
            [self.A[base_len:,:][:,base_len:]]*num_copies,
            format='lil'
        )

        print(f'new matrix = {time.time() - s}')
        s = time.time()

        # fill in the in and out edges
        # this step is O(in_edges * out_edges) <<< O(m)
        i = 0
        for in_edge in in_edges:
            for out_edge in out_edges:

                base_offset = base_len + i*spec_len
                in_offset = base_offset+in_edge[0]
                out_offset = base_offset+out_edge[1]

                S[in_offset,in_edge[1]] = in_graph[in_edge[0],in_edge[1]]
                S[out_edge[0],out_offset] = out_graph[out_edge[0],out_edge[1]]

                i += 1

        S = S.tocsr()

        print(f'fill in edges = {time.time() - s}')
        s = time.time()
        

        # update all the attributes
        # labels = list(self.labeler.values()) + [None]*(num_copies-1)*spec_len
        labels = self.labels[:base_len] + [None]*num_copies*spec_len
        for i in range(num_copies):
            for j in range(spec_len):
                copy_index = base_len+i*spec_len+j
                orig_index = base_len+j
                labels[copy_index] = self.labels[orig_index] + f'.{i+1}'
        

        return Graph(S,labels)
        