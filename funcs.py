import numpy as np
from datetime import datetime
from scipy.linalg import lapack

# Credits to https://stackoverflow.com/a/58719188/12337518
inds_cache = {}

def upper_triangular_to_symmetric(ut):
    n = ut.shape[0]
    try:
        inds = inds_cache[n]
    except KeyError:
        inds = np.tri(n, k=-1, dtype=bool)
        inds_cache[n] = inds
    ut[inds] = ut.T[inds]

# Faster than solving an LU system each time
def fast_positive_definite_inverse(m):
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise ValueError('dpotrf failed on input {}'.format(m))
    inv, info = lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError('dpotri failed on input {}'.format(cholesky))
    upper_triangular_to_symmetric(inv)
    return inv


def show_progress(res):
    now = datetime.now().strftime("%H:%M:%S")
    print(now, res)







