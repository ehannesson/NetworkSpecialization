import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import autograd as ag
import autograd.numpy as anp
from numba import jit


################################ WORK TO BE DONE ##############################
"""
For large matrices we get memory errors, I've found that the upper bound on the
number of nodes it around 13,000, after that the ndarray is no longer a good
tool to use to store the information about the network. Thankfully, these
graphs we are working with are fairly sparse so they should be able to be
implemented in a sparse matrix.
"""
###############################################################################


class DirectedGraph:

    """
    Creates a directed graph that is meant to represent a dynamical network,
    both it's structure and dynamics

    Attributes:
        A (Square, ndarray): the adjecency matrix of a directed graph where
            A(i,j) is node i receiving from node j
        n (int): the number of nodes in the graph
        dynamics (tuple (a,f)):
            a (float): effect of node on itself
            f (nxn matrix valued function): functional relation between each
                of the nodes
        labels (list(str)): list of labels assigned to the nodes of the graph
        labeler (dict(int, str)): maps indices to labels
        indexer (dict(str, int)): maps labels to indices
        colors (dict(int, list(int))): cluster dictionary

    Methods:
        specialize()
        iterate()
        structural_eigen_centrality()
        eigen_centrality()
        detect_sync()
        spectral_radius()
        network_vis()
        coloring()
    """

    def __init__(self, A, dynamics, labels=None):
        """
        Parameters:
            A ((n,n) ndarray): The asjecency matrix to a directed graph where
                A[i,j] is node i receiving from node j
            Labels (list(str)): labels for the nodes of the graph, defaults to
                0 indexing
        """

        n,m = A.shape
        if n != m :
            raise ValueError('Matrix not square')
        if np.diag(A).sum() != 0:
            raise ValueError('Some nodes have self edges')
        # Determine how that labels will be created
        # defaults to simple 0 indexing
        if type(labels) == type(None):
            labels = [f'{i}' for i in range(n)]
        elif (
            type(labels) != type(list()) or
            len(labels) != n or
            type(labels[0]) != type(str())
        ):
            raise ValueError('labels must be a string list of length n')

        self.A = A
        self.n = n
        self.dynamics = dynamics
        self.indices = np.arange(n)
        self.labeler = dict(zip(np.arange(n), labels))
        self.indexer = dict(zip(labels, np.arange(n)))
        # we use the original indexer when we look at dynamics on the network
        # this dict doesn't change under specialization
        self.original_indexer = self.indexer.copy()
        self.colors = dict()
        self.trivial_clusters = set()
        self.nontrivial_nodes = set(self.indices)



    def origination(self, i):
        """
        Returns the original index, associated with the matrix valued dynamics
        function, of a given node index

        Parameters:
            i (int): the current index of a given node in self.A
        Returns:
            o_i (int): the original index of i
        """

        label = self.labeler[i]
        # find the first entry in the node's label
        temp_ind = label.find('.')
        # if it is not the original label we use temp_ind
        if temp_ind != -1:
            label = label[:temp_ind]
        return self.original_indexer[label]

    def set_dynamics(self):
        """
        Using a matrix valued function, set the dynamics of the network

        Implicit Parameters:
            f (nxn matrix valued function): this discribes the independent
                influence the jth node has on the ith node it will use the
                format for the position i,j in the matrix, node i receives
                from node j
        Returns:
            F (n-dimensional vector valued function): this is a vector valued
                function that takes in an ndarray of node states and returns
                the states of the nodes at the next time step
        """

        def create_component_function(A,f,i, o_i):
            """Returns the ith component function of the network"""

            def compt_func(x):
                return np.sum([self.A[i,j]*f[o_i,self.origination(j)](x[j]) for j in range(self.n)])
            return compt_func

        # here we unpack the different parts of the dynamic function
        a,f = self.dynamics

        F = [None]*self.n
        for i in range(self.n):
            o_i = self.origination(i)
            # for each node we create a component function that works much like
            # matrix multiplication
            F[i] = create_component_function(self.A, f, i, o_i)
        # we return a vector valued function that can be used for iteration
        return a, F

    def iterate(
            self, iters, initial_condition,
            graph=False, save_img=False, title=None):
        """
        Model the dynamics on the network for iters timesteps given an intial
        condition

        Parameters
            iters (int): number of timsteps to be simulated
            initial_condition (ndarray): initial conditions of the nodes
            graph (bool): will graph states of nodes over time if True
        Returns:
            x (ndarray): the states of each node at every time step
        """

        # grab the iterative funciton
        a, f = self.set_dynamics()
        def G(t): return [a[self.origination(k)](t[k]) + f[k](t) for k in range(self.n)]

        # initialize an array to be of length iters
        t = [None]*iters
        # set the first entry to be the initial condition
        t[0] = initial_condition
        # the ith entry is the state of the network at time step i
        for i in range(1,iters):
            t[i] = G(t[i-1])
        t = np.array(t)

        # for graphing or saving a graph
        if graph or save_img:
            domain = np.arange(iters)
            for i in range(self.n):
                plt.plot(domain, t[:,i], label=self.labeler[i], lw=2)
            plt.xlabel('Time')
            plt.ylabel('Node Value')
            plt.title('Network Dynamics')
            plt.legend()
            if save_img:
                plt.savefig(title)
            if graph:
                plt.show()
        return t

    def iterate_with_perturbations(
            self, iters, initial_condition, perterbations,
            graph=False, save_img=False, title=None):
        """
        Model the dynamics on the network for iters timesteps given an intial
        condition

        Parameters
            iters (int): number of timsteps to be simulated
            initial_condition (ndarray): initial conditions of the nodes
            perterbations (tuple(timesteps, scale)):
                timesteps (list): time steps to perterb the system at
                scalar (float): scalar used to scale the random perterbations
            graph (bool): will graph states of nodes over time if True
        Returns:
            x (ndarray): the states of each node at every time step
        """

        # grab the iterative funciton
        a, f = self.set_dynamics()
        def G(t): return [a[self.origination(k)](t[k]) + f[k](t) for k in range(self.n)]

        # initialize an array to be of length iters
        t = [None]*iters
        # set the first entry to be the initial condition
        t[0] = initial_condition
        # the ith entry is the state of the network at time step i
        for i in range(1,iters):
            if i in perterbations[0]:
                t[i] = t[i-1] + np.random.random(self.n)*perterbations[1]
            else:
                t[i] = G(t[i-1])
        t = np.array(t)

        # for graphing or saving a graph
        if graph or save_img:
            domain = np.arange(iters)
            for i in range(self.n):
                plt.plot(domain, t[:,i], label=self.labeler[i], lw=2)
            plt.xlabel('Time')
            plt.ylabel('Node Value')
            plt.title('Network Dynamics')
            plt.legend()
            if save_img:
                plt.savefig(title)
            if graph:
                plt.show()
        return t


    def specialize(self, base, verbose=False,recolor=True):
        """
        Given a base set, specialize the adjacency matrix of a network

        Parameters:
            base (list, int or str): list of base nodes, the other nodes will
                become the specialized set
            verbose (bool): print out key information as the code executes
        """

        # if the base was given as a list of nodes then we convert them to the
        # proper indexes
        if type(base[0]) == str:
            for i, k in enumerate(base):
                base[i] = self.indexer[k]
        elif type(base[0]) != int:
            if not np.issubdtype(base[0], np.integer):
                raise ValueError('base set must be either a list of labels'
                    'or a list of indices')
        if type(base) != list:
            base = list(base)
        if len(base) > self.n:
            raise ValueError('base list is too long')

        base_size = len(base)
        # permute the matrix so the base set comes first
        self._base_first(base)

        # initialize the new specialized matrix as a diagonal matrix with the
        # base set in the first slot
        B = self.A[:base_size, :base_size].copy()
        diag = [B]

        # find the strongly connected components of the network
        smallA, comp = self._compress_graph(base_size)

        pressed_paths = self._find_paths_to_base(smallA, base_size, comp)

        n_nodes = base_size
        links = []

        # we create this diag labeler which will associate an index of the
        # list diag to a strongly connected component set
        diag_labeler = {}
        i = 1

        for path in pressed_paths:
            components = [comp[k] for k in path]
            paths = self._path_combinations(components)
            comp_to_add = [self.A[[self.indexer[k] for k in c], :][:, [self.indexer[k] for k in c]] for c in components[1:-1]].copy()

            for p in paths:
                for compt in components[1:-1]:
                    diag_labeler[i] = compt
                    i += 1
                diag += comp_to_add
                links += self._link_adder(p, n_nodes, components)
                n_nodes += sum(map(len, comp_to_add))

        # we update the labeler to correctly label the newly created nodes
        step = base_size
        for i in range(1,len(diag)):
            comp_len = diag[i].shape[0]
            for k in range(comp_len):
                self.labeler[step + k] = diag_labeler[i][k] + f'.{i}'
            step += comp_len

        self._update_indexer()

        if verbose:
            print(f'This is the original matrix:\n{self.A}\n')

        # create the new specialized matrix
        S = la.block_diag(*diag)
        for l in links: S[l] = 1
        self.A = S
        self.indices = np.arange(n_nodes)
        self.n = self.A.shape[0]

        if verbose:
            print(f'Strongly connected components:\n {comp}\n')
            temp_paths = []
            for path in pressed_paths:
                temp_paths.append([comp[k] for k in path])
            print(f'Paths from base node to base node:\n {temp_paths}\n')
            print(f'Number of nodes in the specialized matrix:\n {n_nodes}\n')

        # the coloring must be set to none
        self.colors = dict()

        if recolor:
            self.coloring()

        return

    def _update_indexer(self):
        """
        This function assumes that self.labeler is correct in its labeling:
        it reassigns the labels to the correct indices
        """

        indices = self.labeler.keys()
        labels = self.labeler.values()
        self.indexer = dict(zip(labels, indices))
        return

    def _update_labeler(self):
        """
        This function assumers that self.indexer is correct in its indexing:
        it reassigns the indices to the correct labels
        """

        labels = self.indexer.keys()
        indices = self.indexer.values()
        self.labeler = dict(zip(indices, labels))
        return

    def _base_first(self, base):
        """
        Permutes the A matrix so that the base set corrosponds to the beginning
        set of rows and columns in A

        Parameters:
            base (list, int): a list of the indices of the base set

        Returns:
            None
        """

        # find the indices of the strongly connected set and append them to
        # the end of the base set to for the permutation of A


        to_specialize = [i for i in self.indices if i not in base]
        permute = base + to_specialize
        # Change the labeler to reflect the permutation and update the indexer
        self.labeler = {i : self.labeler[permute[i]] for i in range(self.n)}
        self._update_indexer()

        # This will rearrange the indices to put the base set first
        pA = self.A[permute,:]
        pA = pA[:,permute]

        self.A = pA
        return

    def _compress_graph(self, base_size):
        """
        Creates a new matrix smallA that is the compressed adjacency matrix of
        A, each strongly connected component is represented as a single node

        Parameters:
            base_size (int): number of nodes in the base set

        Returns:
            smallA (ndarray, square): compressed adjacency matrix of A
            comp (dict, int: list(str)): a labling dictionary maping each node
                of the compressed graph to the set of nodes it represents
        """

        # find the strongly connected components
        spec = self.A[base_size:, base_size:]

        # we use nx to create a directed graph of this part
        spec_graph = nx.DiGraph(spec.T)

        # use nx to find the strongly connected components of specG
        SCComp = [np.array(list(c)) for c in nx.strongly_connected_components(spec_graph)]
        num_comp = len(SCComp)
        N = base_size + num_comp

        # comp will be a dictionary mapping base nodes to components
        # comp will keep track of the labels of the components
        comp = {}
        for i in range(base_size):
            comp[i] = [self.labeler[i]]

        for i in range(num_comp):
            temp1 = SCComp[i]
            comp[i+base_size] = [self.labeler[k+base_size] for k in temp1]

        # smallA will for our compressed A matrix lumping together all the
        # strongly connected components
        smallA = np.zeros((N,N))
        smallA[:base_size, :base_size] = self.A[:base_size, :base_size]

        for i in range(base_size, N):
            i_indices = [self.indexer[k] for k in comp[i]]

            smallA[:base_size, i] = (self.A[:base_size, i_indices].sum(axis=1) != 0)*1.
            smallA[i, :base_size] = (self.A[i_indices, :base_size].sum(axis=0) != 0)*1.

            for j in range(base_size, i):
                j_indices = [self.indexer[k] for k in comp[j]]

                smallA[i,j] = (not (self.A[i_indices, :][:, j_indices] == 0).all())*1.
                smallA[j,i] = (not (self.A[j_indices, :][:, i_indices] == 0).all())*1.
        return smallA, comp

    def _find_paths_to_base(self, smallA, base_size, comp):
        """
        Finds all the paths between the base nodes that pass through the
        specialization set in the compressed graph

        Parameters:
            smallA (ndarray): a compressed adjecency matrix
            base_size (int): number of nodes in the base set

        Returns:
            pressed_paths (list, list(str)): list of paths that pass through
                the specialization set
        """

        _, N = smallA.shape
        pressed_paths = []

        # find the path between every pair of nodes
        for b1 in range(base_size):
            for b2 in range(base_size):
                # remove all other base ndes from the graph so we only find
                # paths through the specialization set
                if b1 == b2:
                    # in this case we look for cycles
                    mask = [b1] + list(range(base_size, N))
                    new_size = len(mask) + 1
                    reducedA = np.zeros((new_size, new_size))
                    reducedA[:-1, :-1] = smallA[mask,:][:, mask]
                    # remove in going edges from the base node and add to new
                    # node
                    reducedA[-1,:] = reducedA[0,:]
                    reducedA[0,:] = np.zeros(new_size)
                    G = nx.DiGraph(reducedA.T)
                    # find paths from the base node to the new node
                    # which is the same as finding cycles in this case
                    paths = list(nx.all_simple_paths(G,0,new_size-1))

                else:
                    mask = [b1,b2] + list(range(base_size, N))
                    reducedA = smallA[mask,:][:,mask]
                    # remove base node interations
                    reducedA[:2,:2] = np.zeros((2,2))
                    G = nx.DiGraph(reducedA.T)
                    paths = list(nx.all_simple_paths(G,0,1))

                # process the paths
                for p in paths:
                    if p != []:
                        if b1 == b2:
                            p = np.array(p) + base_size - 1
                        else:
                            p = np.array(p) + base_size - 2
                        p[[0,-1]] = [b1,b2]
                        pressed_paths.append(p)
        return pressed_paths

    def _path_combinations(self, components):
        """
        Given a path through the connected components of A, find every unique
        combination of edges between the components that can be followed to
        complete the given path

        Parameters:
            components (list, ):
        """

        link_opt = []
        path_length = len(components)
        # n_nodes will keep track of the number of nodes in each branch
        n_nodes = 1

        for i in range(path_length - 1):
            _i = [self.indexer[k] for k in components[i+1]]
            _j = [self.indexer[k] for k in components[i]]
            rows, cols = np.where(self.A[_i,:][:,_j] == 1)

            if i == 0:
                cols += [self.indexer[k] for k in components[0]]
                rows += n_nodes
            elif i == path_length - 2:
                rows += [self.indexer[k] for k in components[-1]]
                cols += n_nodes - len(components[i])
            else:
                rows += n_nodes
                cols += n_nodes - len(components[i])
            edges = zip(rows,cols)
            n_nodes += len(components[i+1])
            link_opt.append(edges)

        all_paths = [list(P) for P in itertools.product(*link_opt)]
        return all_paths

    def _link_adder(self, path, n_nodes, components):
        """
        Produces the link needed to add a branch of stringly connected
        components to a graph with n_nodes

        Parameters:
            path (list, tuple): edges between component nodes
            n_nodes (int): number of nodes in the original graph
            components (list, list()):
        """

        links = []
        path_length = len(path)

        for i in range(path_length):
            if i == 0:
                links.append((path[i][0] + n_nodes - 1, path[i][1]))
            elif i == path_length - 1:
                links.append((path[i][0], path[i][1] + n_nodes - 1))
            else:
                links.append((path[i][0] + n_nodes-1, path[i][1] + n_nodes-1))

        return links



    def stability_matrix(self):
        """
        Returns:
            (ndarray): the stability matrix of the network where the i,j entry
                is the supremum of the partial ith function with respect to
                the jth argument, which is the derivative of the i,jth entry
                of self.dynamics[1]
        """

        # extract the functions that determine the dynamics
        a,f = self.dynamics
        Df = np.zeros((self.n,self.n))
        domain = np.linspace(-10,10,50000)

        # since the i,j entry of the stability matrix is the absolute
        # supremum of the partial of the ith function with respect to
        # the the jth variable we loop through the nxn Df matrix and
        # look the the derivatives of the entries of the functional
        # matrix
        for i in range(self.n):
            o_i = self.origination(i)
            for j in range(self.n):
                o_j = self.origination(j)

                # if i == j we consider the self edge case
                if i == j:
                    # find the negative of the function's derivative
                    func = a[o_i]
                    def _df(x): return -1.*ag.elementwise_grad(func)(x)
                    _range = _df(domain)
                    Df[i,j] = np.max(np.abs(_range))
                    # result = opt.minimize_scalar(_df)
                    # if the function didn't minimize we raise a warning
                    # if not result['success']:
                    #     raise RuntimeWarning(
                    #         'one of the derivatives did not minimize')
                    # Df[i,j] = result['fun']

                # since there is never an edge between nodes that share
                # an origination we set this value to 0
                elif o_i == o_j:
                    Df[i,j] = 0

                # here we consider the edges between nodes
                else:
                    func = f[o_i,o_j]
                    def _df(x): return -1.*ag.elementwise_grad(func)(x)
                    _range = _df(domain)
                    Df[i,j] = np.max(np.abs(_range))
                    # plt.plot(domain,_range)
                    # plt.show()
                    # result = opt.minimize_scalar(_df)
                    # if not result['success']:
                    #     raise RuntimeWarning(
                    #         'one of the derivatives did not minimize')
                    # Df[i,j] = result['fun']

        return Df

    def eigen_centrality(self):
        """
        Returns:
            (dict):
                (keys): labels of the nodes
                (values): the eigen centrality of that node
        """

        # we shift the matrix to find the true dominant eigen value
        B = self.stability_matrix() + np.eye(self.n)
        eigs = la.eig(B)
        i = np.argmax(eigs[0])
        # we then extract the eigen vector associated to this value
        p = eigs[1][:,i]
        # normalize the vector so it sums to 1, i.e. turn it into a
        # probability vector
        p /= p.sum()
        ranks = {self.labeler[i]: np.real(p[i]) for i in range(self.n)}
        return ranks

    def spectral_radius(self):
        """
        Returns:
            (float): the spectral radius of the network based on the stability
                matrix
        """

        Df = self.stability_matrix()
        eigs = la.eig(Df)
        # find the eigen value with largest modulus, which is the spectral
        # radius
        return np.max(np.abs(eigs[0]))

    def structural_eigen_centrality(self):
        """
        Returns:
            (dict):
                (keys): labels of the nodes
                (values): the associated eigencentrality
        """

        # we shift the matrix to find the true dominant eigen value
        B = self.A + np.eye(self.n)
        eigs = la.eig(B)
        i = np.argmax(eigs[0])
        # we then extract the eigen vector associated to this value
        p = eigs[1][:,i]
        # normalize the vector so it sums to 1
        # i.e. turn it into a probability vector
        p /= p.sum()
        ranks = {self.labeler[i]: np.real(p[i]) for i in range(self.n)}
        return ranks

    def in_degree(self):
        labels = [self.labeler[i] for i in range(self.n)]
        return dict(zip(labels, np.sum(self.A, axis=1)))


    def detect_sync(self, iters=80, sync_tol=1e-2, otol=1e-1):
            """
            Determines which nodes in a network synchronize.

            Parameters:
                G (DirectedGraph): The DirectedGraph of interest
                iter_matrix (ndarray): mxn array containing the values of m
                    dynamic iterations of the n nodes of G
                iters (int): number of iterations to produce iter_matrix
                    (for use when iter_matrix is not explicitly passed in)
                sync_tol (float): tolerance for synchronization
                otol (float): tolerance for stability

            Returns:
                sync_communities (tuple): of the form ((community), bool),
                        where the bool represents if the community is stable
            """

            iter_matrix = self.iterate(iters, np.random.random(self.n)*10)
            # number of nodes in network
            n = iter_matrix.shape[1]

            # only check the last few rows for convergence
            A = iter_matrix[-5:, :]

            # tracks which nodes/communities have synchronized
            sync_communities = []
            sync_nodes = []

            for node in range(n):
                if node in sync_nodes:
                    continue
                else:
                    # sync_ind is a list of nodes synchronized with 'node'
                    _A = (A.T - A[:, node]).T
                    is_sync = np.all(np.isclose(_A, 0, atol=sync_tol), axis=0)
                    ind = np.where(is_sync)[0]
                    sync_ind = list(ind)
                    # track which nodes have already synchronized so we don't
                    # double check
                    sync_nodes.extend(np.ravel(sync_ind))
                    # track communities of synchronization
                    maping_func = lambda x: self.labeler[x]
                    sync_tup = tuple(map(maping_func, sync_ind))
                    is_close = np.isclose(A[:,node] - A[-1,node], 0, atol=otol)
                    is_conv = np.all(is_close)
                    sync_communities.append((sync_tup, is_conv))

            return sync_communities


    def network_vis(
            self, use_eqp=False, iter_matrix=False, spec_layout=False,
            lin=False, lin_dyn=None, title="Network Visualization",
            save_img=False, filename='network_viz',show=True):
        """
        Creates a visualization of the network G
        Parameters:
            G (DirectedGraph): A DirectedGraph object to visualize
            iter_matrix (ndarray): allows you to explicitly pass in the
                dynamics matrix of G. If left as False, then we run the
                dynamics simulation here
            spec_layout (bool): if False will display with random layout,
                otherwise it will use the spectral layout option
            lin (bool): Adds edge labels if the dynamics are linear
            lin_dyn (ndarray): the linear dynamics array
            title (str): Title of visualization
            save_img (bool): if True, saves the visualization with name
                'filename'
            filename (str): filename
        """

        if use_eqp:
            if self.colors == dict():
                self.coloring()
            self.colors
            group_dict = {}
            for color in self.colors.keys():
                for node in self.colors[color]:
                    group_dict[self.labeler[node]] = color

        else:
            # find synchronized communities
            communities = self.detect_sync(iters=80)

            # create a dictionary mapping each node to its community
            group_dict = {}
            for i in range(len(communities)):
                for node in communities[i][0]:
                    group_dict[node] = i

        # create (and relabel) a networkx graph object
        nxG = nx.relabel.relabel_nodes(nx.DiGraph(self.A.T), self.labeler)

        # set community membership as an attribute of nxG
        nx.set_node_attributes(nxG, group_dict, name='community')

        # list of community number in order of how the nodes are stored
        colors = [group_dict[node] for node in nxG.nodes()]

        plt.figure()

        if lin:
            # for display of edge dynamics (linear dynamics only)
            edge_weights = nx.get_edge_attributes(nxG, 'weight')
            # edit the edge weights to be the correct dynamics
            for edge in edge_weights:
                i = self.origination(self.indexer[edge[1]])
                j = self.origination(self.indexer[edge[0]])
                edge_weights[edge] = lin_dyn[i, j]



        if spec_layout:
            # draw the network
            nx.draw_networkx(nxG, pos=nx.drawing.spectral_layout(nxG),
                node_size=1000, arrowsize=20, node_color=colors,
                cmap=plt.cm.Set3)
            if lin:
                # add edge weights
                nx.draw_networkx_edge_labels(nxG,
                    nx.drawing.spectral_layout(nxG), edge_labels=edge_weights)

        else:
            # draw the network
            nx.draw_networkx(nxG, pos=nx.drawing.layout.planar_layout(nxG),
                node_size=1000, arrowsize=20, node_color=colors,
                cmap=plt.cm.Set3)
            if lin:
                # add edge weights
                nx.draw_networkx_edge_labels(nxG,
                    nx.drawing.layout.planar_layout(nxG),
                    edge_labels=edge_weights)

        plt.title(title)
        if save_img:
            plt.savefig(filename)
        if show:
            plt.show()

    def coloring(self):
        """
        This method uses an algorithm called input driven refinement that will
        find the unique coarsest equitable partition of a the graph associated
        with the network. This partition is based only on the structure of the
        graph, and is not based on other functional elements of the system.
        Returns:
            dict(int: list): a partition where each of the keys represents a
                unique color and the associated list a list of indices that
                are in the cluster
        """

        #helper function for input driven refinement
        # @jit
        def _refine(color_dict):
            # initialize the lists and dictionaries that we need for the input
            # driven refinement

            final_colors = {}
            new_clusters = set()
            temp_new_clusters = []
            temp_trivial_clusters = set()
            # for every cluster we examine the inputs from every other cluster
            for color1 in color_dict.keys():
                for color2 in color_dict.keys():
                    # create a sub graph that is only of the
                    # two colors in question
                    sub_graph = self.A[color_dict[color1]][:,color_dict[color2]]
                    # the row sums will show the number of inputs
                    # from color2 to color1
                    inputs = np.sum(sub_graph, axis=1)
                    input_nums = set(inputs)
                    for num in input_nums:
                        cluster = np.where(inputs == num)[0]
                        # use relative indexing to find the correct nodes
                        cluster = set(color_dict[color1][cluster])
                        if cluster not in temp_new_clusters:
                            temp_new_clusters.append(cluster)
                            if len(cluster) == 1:
                                self.trivial_clusters.update(cluster)
                                temp_trivial_clusters.add(tuple(cluster))

            # remove trivial clusters from potential clusters\
            temp_new_clusters = {tuple(cluster - self.trivial_clusters) for cluster in temp_new_clusters}
            
            try:
                temp_new_clusters.remove(tuple())
            except:
                pass
            
            # for every node we find the smallest cluster containing that
            # node, and that is the cluster we submit for the new coloring
            for node in self.nontrivial_nodes:
                potential = None
                len_potential = np.inf
                for cluster in temp_new_clusters:
                    if node in cluster and len(cluster) < len_potential:
                        potential = cluster#.copy()
                        len_potential = len(potential)

                if potential not in new_clusters and potential is not None:
                    new_clusters.add(potential)

            # add trivial clusters back
            new_clusters.update(temp_trivial_clusters)

            # remove trivial clusters from nontrivial nodes for temporal improvement
            self.nontrivial_nodes -= self.trivial_clusters
            for i, cluster in enumerate(new_clusters):
                final_colors[i] = np.array(list(cluster))
            return final_colors


        # we begin by coloring all every node the same color
        next_colors = {0 : self.indices.copy()}

        # reset the attributes that influence the outcome of the algorithm
        self.trivial_clusters = set()
        self.nontrivial_nodes = set(self.indices)

        # we then iteratively apply input driven refinement to coarsen
        refine = True
        counter = 0
        while refine:
            counter += 1
            colors = _refine(next_colors)
            #if there are no new colors then the refinement is equivalent\
            next_colors = _refine(colors)
            if len(colors.keys()) == len(next_colors.keys()):
                refine = False
        self.colors = colors


    def color_checker(self):
        for color1 in self.colors.values():
            for color2 in self.colors.values():
                x = np.sum(self.A[color1][:,color2],axis=1)
                if not (x==x[0]).all():
                    return False
        return True
