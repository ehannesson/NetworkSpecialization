cimport cython
from cython.parallel import prange
from cython import parallel
from libc.stdlib cimport abort, malloc, free
cimport openmp


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int contains(int[:] arr, int val) nogil:
    cdef int i
    for i in range(arr.shape[0]):
        if val == arr[i]:
            return 1
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int count_hit(int[:] neighbor_nodes, int[:] color_class_nodes) nogil:
    """
    Counts how many times a node is hit by a color class.

    Parameters
    ----------
    neighbor_nodes : integer array
        Indices corresponding to the neighbors of the node in question.

    color_class_nodes : integer array
        Indices corresponding to the members of the color class in question.
    """

    cdef int count, neighbor, colored_node
    count = 0

    # iterate through the node's neighbors
    for neighbor in range(neighbor_nodes.shape[0]):
        # check if any node in the color class is this neighbor
        for colored_node in range(color_class_nodes.shape[0]):
            if neighbor_nodes[neighbor] == color_class_nodes[colored_node]:
                count = count + 1
                break

    return count


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.unraisable_tracebacks(False)
cdef int check_class(int class_one_ind, int class_two_ind, int[:] indptr, int[:] inds, int[:] c_indptr, int[:] c_inds) nogil:
    """
    Given two color classes, checks that every node in the first one is hit by
    the same number of nodes in the second one.

    Parameters
    ----------
    class_one_ind : integer

    class_two_ind : integer
    """

    cdef:
        int[:] class_one = c_indptr[c_inds[class_one_ind]:c_inds[class_one_ind+1]]
        int[:] class_two = c_indptr[c_inds[class_two_ind]:c_inds[class_two_ind+1]]

        int hit = count_hit(indptr[inds[class_one[0]]:inds[class_one[1]]], class_two)
        int n
        int size = class_one.shape[0]

    for n in range(1, size):
        if hit != count_hit(indptr[inds[class_one[n]]:inds[class_one[n+1]]], class_two):
            return 0
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int check_ep(int[:] indptr, int[:] inds, int[:] c_inds, int[:] c_indptr, int num_threads):
    """
    Verifies the proposed equitable partition is, in fact, equitable.

    Parameters
    ----------
    indptr : ndarray(np.int32)
        indptr array for the adjacency matrix when stored in CSR format
    inds : ndarray(np.int32)
        Column Array for the adjacency matrix when stored in CSR format
    c_inds : ndarray(np.int32)
        Similar to the indptr list of a CSR matrix, c_inds[i] contains the first
        index at which nodes in color class i will be found in c_indptr
    c_indptr : ndarray(np.int32)
        List of nodes, grouped by color as specified by c_inds.
    num_threads : int
        Number of cpu threads to use.
    """
    cdef int c1, c2
    cdef int n = c_inds.shape[0]

    for c1 in prange(n, nogil=True):
        for c2 in range(n):
            if check_class(c1, c2, indptr, inds, c_indptr, c_inds) == 0:
                return 0
    return 1




# neighbors_of_node_i = indptr[inds[i]:inds[i+1]]
# nodes_in_color_class_j = c_indptr[c_inds[j]:c_inds[j+1]]




# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef int ep_check(int[:] indptr, int[:] inds, int[:] c_inds, int[:] c_indptr, int num_threads):
#     """
#     Parameters
#     ----------
#     indptr : ndarray(np.int32)
#         indptr array for the adjacency matrix when stored in CSR format
#     inds : ndarray(np.int32)
#         Column Array for the adjacency matrix when stored in CSR format
#     c_inds : ndarray(np.int32)
#         Similar to the indptr list of a CSR matrix, c_inds[i] contains the first
#         index at which nodes in color class i will be found in c_indptr
#     c_indptr : ndarray(np.int32)
#         List of nodes, grouped by color as specified by c_inds.
#     num_threads : int
#         Number of cpu threads to use.
#     """
#     cdef int n = c_inds.shape[0]
#     cdef int thread_id
#
#
#     with nogil, parallel.parallel():
#         # setup local buffer pointers
#         local_buf = <int *> malloc(sizeof(int) * num_threads)
#         local_buf0 = <int *> malloc(sizeof(int) * num_threads)
#
#         if local_buf is NULL or local_buf0 is NULL:
#             abort()
#
#         # initialize buffers to zeros
#         for i in range(num_threads):
#             local_buf[i] = 0
#             local_buf0[i] = 0
#
#         for cind1 in prange(n, schedule='guided'):
#             thread_id = parallel.threadid()
#             for cind2 in range(n):
#                 for c1 in range(c_inds[cind1]):
#                     if c1 == 0:
#                         local_buf[thread_id] = 0
#                     a = colors[cind1, :][c1]
#                     for c2 in range(c_inds[cind2]):
#                         b = colors[cind2, :][c2]
#                         local_buf[thread_id] += A[a, b]
#
#                     if c1 == 0:
#                         local_buf0[thread_id] = local_buf[thread_id]
#                     else:
#                         if local_buf0[thread_id] != local_buf[thread_id]:
#                             return 0
#
#         free(local_buf)
#         free(local_buf0)
#
#     return 1






# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef int ep_check(int[:, ::1] A, int[:, ::1] colors, int[:] c_inds,
#                     int num_threads):
#
#     cdef int n = c_inds.shape[0]
#     cdef int[:] class1, class2
#     cdef int cind1, cind2, c1, c2, a, b, i
#     cdef int thread_id
#
#
#     with nogil, parallel.parallel():
#         local_buf = <int *> malloc(sizeof(int) * num_threads)
#         local_buf0 = <int *> malloc(sizeof(int) * num_threads)
#         # local_c1 = <int *> malloc(sizeof(int) * )
#         # class1 = <int *> malloc(sizeof(int) * num_threads * )
#
#         if local_buf is NULL or local_buf0 is NULL:
#             abort()
#
#         # setup local buffer
#         for i in range(num_threads):
#             local_buf[i] = 0
#             local_buf0[i] = 0
#
#         for cind1 in prange(n, schedule='guided'):
#             thread_id = parallel.threadid()
#             for cind2 in range(n):
#                 for c1 in range(c_inds[cind1]):
#                     if c1 == 0:
#                         local_buf[thread_id] = 0
#                     a = colors[cind1, :][c1]
#                     for c2 in range(c_inds[cind2]):
#                         b = colors[cind2, :][c2]
#                         local_buf[thread_id] += A[a, b]
#
#                     if c1 == 0:
#                         local_buf0[thread_id] = local_buf[thread_id]
#                     else:
#                         if local_buf0[thread_id] != local_buf[thread_id]:
#                             return 0
#
#         free(local_buf)
#         free(local_buf0)
#
#     return 1
