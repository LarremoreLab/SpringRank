import numpy as np
from scipy.sparse import spdiags, csr_matrix, eye
from scipy.optimize import brentq
import scipy.sparse.linalg
import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter('ignore', SparseEfficiencyWarning)

class SpringRank:
    """
    A class implementation of the SpringRank algorithm for computing hierarchical rankings
    from directed networks.
    
    Parameters
    ----------
    alpha : float, default=0
        Regularization parameter. If 0, uses Lagrange multiplier approach.
        If > 0, performs L2 regularization.
        
    Attributes
    ----------
    ranks_ : array-like of shape (n_nodes,)
        The computed ranks for each node after fitting
    is_fitted_ranks_ : bool
        Whether the model has been fitted
    """
    
    def __init__(self, alpha=0):
        self.alpha = alpha
        self.ranks = None
        self.is_fitted_ranks_ = False
        self.beta = None
        self.is_fitted_beta_ = False
        self.A = None

    def fit(self, A):
        """
        Compute the SpringRank solution for the input adjacency matrix.
        
        Parameters
        ----------
        A : array-like or sparse matrix of shape (n_nodes, n_nodes)
            The adjacency matrix of the directed network
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validation of A
        if not (A.shape[0] == A.shape[1]):
            raise ValueError("Adjacency matrix must be square")
        if scipy.sparse.issparse(A):
            neg_entries = A.data < 0
        else:
            neg_entries = A < 0
        if neg_entries.any():
            raise ValueError("Adjacency matrix cannot contain negative entries")

        self.A = A

        if self.alpha == 0:
            self.ranks = self._solve_springrank()
        else:
            self.ranks = self._solve_springrank_regularized()
            
        self.is_fitted_ranks_ = True

        return self
    
    def _solve_springrank(self):
        """Implementation of non-regularized SpringRank"""
        N = self.A.shape[0]
        k_in = np.array(self.A.sum(axis=0))
        k_out = np.array(self.A.sum(axis=1).transpose())

        # form the graph laplacian
        operator = csr_matrix(
            spdiags(k_out + k_in, 0, N, N) - self.A - self.A.transpose()
        )

        # form the operator A (from Ax=b notation)
        # note that this is the operator in the paper, but augmented
        # to solve a Lagrange multiplier problem that provides the constraint
        operator.resize((N + 1, N + 1))
        operator[N, 0] = 1
        operator[0, N] = 1

        # form solution vector b (from Ax=b notation)
        solution_vector = np.append((k_out - k_in), np.array([0])).transpose()

        # solve system
        ranks = scipy.sparse.linalg.bicgstab(
            scipy.sparse.csr_matrix(operator),
            solution_vector
        )[0]

        mean_centered_ranks = ranks[:-1] - np.mean(ranks[:-1])

        return mean_centered_ranks
    
    def _solve_springrank_regularized(self):
        """Implementation of regularized SpringRank"""
        if isinstance(self.A, np.ndarray):
            self.A = csr_matrix(self.A)
            
        N = self.A.shape[0]
        k_in = self.A.sum(axis=0)
        k_out = self.A.sum(axis=1).T

        k_in = spdiags(np.array(k_in)[0], 0, N, N, format="csr")
        k_out = spdiags(np.array(k_out)[0], 0, N, N, format="csr")

        C = self.A + self.A.T
        D1 = k_in + k_out
        B = k_out - k_in
        B = B @ np.ones([N, 1])

        operator = self.alpha * eye(N) + D1 - C
        ranks = scipy.sparse.linalg.bicgstab(operator, B)[0]

        return ranks
    
    @staticmethod
    def _eqs39(beta, s, A):
        """Helper function for inverse temperature calculation
        Memory-efficient version of eqs39 that works with sparse matrices.
        Instead of converting to dense matrix, we iterate over nonzero elements.
        """
        x = 0
        rows, cols = A.nonzero()
        for idx in range(len(rows)):
            i, j = rows[idx], cols[idx]
            a_ij = A[i,j]
            a_ji = A[j,i] 
            x += (s[i] - s[j]) * (a_ij - (a_ij + a_ji) / 
                                 (1 + np.exp(-2 * beta * (s[i] - s[j]))))
        return x
    
    def _get_inverse_temperature(self):
        """Calculate inverse temperature parameter"""
        MLE = brentq(self._eqs39, 0.01, 20, args=(self.ranks, self.A))
        return MLE
    
    def get_beta(self):
        if self.is_fitted_beta_ == False:
            self.beta = self._get_inverse_temperature()
            self.is_fitted_beta_ = True
        return self.beta

    def get_rescaled_ranks(self, target_scale):
        """Rescale ranks using target scale and inverse temperature"""
        if self.is_fitted_beta_ == False:
            self.beta = self._get_inverse_temperature()
            self.is_fitted_beta_ = True
        scaling_factor = 1 / (np.log(target_scale / (1 - target_scale)) / 
                            (2 * self.beta))
        return self.ranks * scaling_factor
    
    def predict(self, ij_pair):
        """Predict probability that i -> j in a pair [i,j]"""
        if not self.is_fitted_ranks_:
            raise ValueError("Call fit before predicting")
        i = ij_pair[0]
        j = ij_pair[1]
        if not (0 <= i < self.A.shape[0] and 0 <= j < self.A.shape[0]):
            raise ValueError(f"Indices {i}, {j} out of bounds for matrix of size {self.A.shape[0]}")
        if self.is_fitted_beta_ == False:
            self.beta = self._get_inverse_temperature()
            self.is_fitted_beta_ = True
        diff = self.ranks[i] - self.ranks[j]
        return 1 / (1 + np.exp(-2 * self.beta * diff))