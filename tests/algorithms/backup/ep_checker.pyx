def ep_check(int[:, :] A, int[:, :] colors, int[:] color_inds):

    cdef int n = color_inds.shape[0]
    cdef int[:] class1, class2
    cdef int temp_sum0, temp_sum, a, b

    for cind1 in range(n):
        class1 = colors[cind1, :]

        for cind2 in range(n):
            class2 = colors[cind2, :]

            for c1 in range(color_inds[cind1]):
                temp_sum = 0
                a = class1[c1]
                for c2 in range(color_inds[cind2]):
                    b = class2[c2]
                    temp_sum += A[a, b]

                if c1 == 0:
                    temp_sum0 = temp_sum
                else:
                    if temp_sum != temp_sum0:
                        return False
    return True
