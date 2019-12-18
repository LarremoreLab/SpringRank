import numpy as np
from numba import jit
from scipy.sparse import spdiags, csr_matrix
from scipy.optimize import brentq
import scipy.sparse.linalg

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)


def get_scaled_ranks(A, scale=0.75):
    """
    params:
    - A: a (square) np.ndarray
    - scale: float, greater than 0 less than 1; scale is the probability 
        that a node x with rank R_x will beat nodes with rank R_x - 1

    returns:
    - ranks, np.array

    TODO:
    - support passing in other formats (eg a sparse matrix)
    """
    ranks = get_ranks(A)
    inverse_temperature = get_inverse_temperature(A, ranks)
    scaling_factor = 1 / (np.log(scale / (1 - scale)) / (2 * inverse_temperature))
    scaled_ranks = scale_ranks(ranks,scaling_factor)
    return scaled_ranks


def get_ranks(A):
    """
    params:
    - A: a (square) np.ndarray

    returns:
    - ranks, np.array

    TODO:
    - support passing in other formats (eg a sparse matrix)
    """
    return SpringRank(A)


def get_inverse_temperature(A, ranks):
    """given an adjacency matrix and the ranks for that matrix, calculates the
    temperature of those ranks"""
    betahat = brentq(eqs39, 0.01, 20, args=(ranks, A))
    return betahat


def scale_ranks(ranks, scaling_factor):
    return ranks * scaling_factor


def csr_SpringRank(A):
    """
    Main routine to calculate SpringRank by solving linear system
    Default parameters are initialized as in the standard SpringRank model

    Arguments:
        A: Directed network (np.ndarray, scipy.sparse.csr.csr_matrix)

    Output:
        rank: N-dim array, indeces represent the nodes' indices used in ordering the matrix A
    """

    N = A.shape[0]
    k_in = np.array(A.sum(axis=0))
    k_out = np.array(A.sum(axis=1).transpose())

    # form the graph laplacian
    operator = csr_matrix(
        spdiags(k_out + k_in, 0, N, N) - A - A.transpose()
    )

    # form the operator A (from Ax=b notation)
    # note that this is the operator in the paper, but augmented
    # to solve a Lagrange multiplier problem that provides the constraint
    operator.resize((N + 1, N + 1))
    operator[N, 0] = 1
    operator[0, N] = 1

    # form the solution vector b (from Ax=b notation)
    solution_vector = np.append((k_out - k_in), np.array([0])).transpose()

    # perform the computations
    ranks = scipy.sparse.linalg.bicgstab(
        scipy.sparse.csr_matrix(operator),
        solution_vector
    )[0]

    return ranks[:-1]


def SpringRank(A, alpha=0):
    """
    Solve the SpringRank system.
    If alpha = 0, solves a Lagrange multiplier problem.
    Otherwise, performs L2 regularization to make full rank.

    Arguments:
        A: Directed network (np.ndarray, scipy.sparse.csr.csr_matrix)
        alpha: regularization term. Defaults to 0.

    Output:
        ranks: Solution to SpringRank
    """

    if alpha == 0:
        rank = csr_SpringRank(A)
    else:
        N = A.shape[0]
        k_in = np.sum(A, 0)
        k_out = np.sum(A, 1)

        C = A + A.T
        D1 = np.diag(k_out + k_in)
        d2 = k_out - k_in
        B = alpha + d2
        A = alpha * np.eye(N) + D1 - C
        A = scipy.sparse.csr_matrix(np.matrix(A))
        rank = scipy.sparse.linalg.bicgstab(A, B)[0]

    return np.transpose(rank)


@jit(nopython=True)
def eqs39(beta, s, A):
    N = A.shape[0]
    x = 0
    for i in range(N):
        for j in range(N):
            if A[i, j] == 0:
                continue
            else:
                x += (s[i] - s[j]) * (A[i, j] - (A[i, j] + A[j, i]) / (1 + np.exp(-2 * beta * (s[i] - s[j]))))
    return x
