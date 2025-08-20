import numpy as np
from scipy.sparse import spdiags, csr_matrix, eye
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
import scipy.sparse.linalg
import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter("ignore", SparseEfficiencyWarning)

class SpringRank:
    """
    A class implementation of the SpringRank algorithm for computing hierarchical rankings
    from directed networks.

    Parameters
    ----------
    alpha : float, default=0
        Regularization parameter. If 0, uses Lagrange multiplier approach.
        If > 0, performs L2 regularization.
    rtol : float, default=1e-05
        Relative tolerance for the iterative rank solver.
    atol : float, default=0.0
        Absolute tolerance for the iterative rank solver.
    inverse_temp_type : str, default="global"
        Type of inverse temperature parameter to use for prediction and rank rescaling. Can be "global" for global inverse temperature (optimizes for conditional log-likelihood sigma_L)
        or "local" for local inverse temperature (optimizes for local accuracy sigma_a).
    inverse_temp_fit_warning : bool, default=True
        Whether to issue a warning when the inverse temperature parameter fitting does not converge.
        If fitting by either root-finding Eq. S39 or minimizing Eq. 12 fails, a warning will be issued and the inverse temperature parameter will default to 20.
    max_beta : float, default=20
        Maximum value for the inverse temperature parameter. 
        If the optimal beta exceeds this value or if the model is fit to a perfect hierarchy, beta will be capped at this value.

    Attributes
    ----------
    ranks_ : array-like of shape (n_nodes,)
        The computed ranks for each node after fitting
    is_fitted_ranks_ : bool
        Whether the model has been fitted
    """

    def __init__(
        self,
        alpha=0,
        rtol=None,
        atol=None,
        inverse_temp_type="global",
        inverse_temp_fit_warning=True,
        max_beta=20
    ):
        assert inverse_temp_type in [
            "global",
            "local",
        ], "inverse_temp_type must be either 'global' or 'local'"

        self.alpha = alpha
        self.ranks = None
        self.is_fitted_ranks_ = False
        self.beta = None
        self.is_fitted_beta_ = False
        self.inverse_temp_type = inverse_temp_type
        self.warn_beta = inverse_temp_fit_warning
        self.max_beta = max_beta
        self.A = None
        self.rtol = 1e-05 if rtol is None else rtol
        self.atol = 0.0 if atol is None else atol

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
            solution_vector,
            rtol=self.rtol,
            atol=self.atol,
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
        ranks = scipy.sparse.linalg.bicgstab(
            operator, B, rtol=self.rtol, atol=self.atol
        )[0]

        return ranks

    @staticmethod
    def _eqs39(beta, s, A):
        """
        Helper function for global inverse temperature optimization.
        Memory-efficient version of Eq S39 that works with sparse matrices.
        Instead of converting to dense matrix, we iterate over nonzero elements.
        """
        x = 0
        rows, cols = A.nonzero()
        for idx in range(len(rows)):
            i, j = rows[idx], cols[idx]
            a_ij = A[i, j]
            a_ji = A[j, i]
            x += (s[i] - s[j]) * (
                a_ij - (a_ij + a_ji) / (1 + np.exp(-2 * beta * (s[i] - s[j])))
            )
        return x

    @staticmethod
    def _eq12_ish(beta, s, A):
        """
        Helper function for local inverse temperature optimization. 
        In the paper, we optimize beta_a to be the value that maximizes sigma_a in Eq 12.
        Equivalently, we can minimize 2M*(1-sigma_a), which is what this _eq12_ish equation is.
        """
        x = 0
        rows, cols = A.nonzero()
        for idx in range(len(rows)):
            i, j = rows[idx], cols[idx]
            a_ij = A[i, j]
            a_ji = A[j, i]
            a_ij_bar = a_ij + a_ji
            x += np.abs(a_ij - a_ij_bar / (1 + np.exp(-2 * beta * (s[i] - s[j]))))
        return x

    def _get_inverse_temperature(self):
        """
        Calculates the temperature parameter used for prediction and rank rescaling.
        If self.inverse_temp_type = "global", computes the global inverse temperature parameter beta_L that maximizes the conditional log-likelihood sigma_L (Eq. 13).
        If self.inverse_temp_type = "local", computes the local inverse temperature parameter beta_a that maximizes the local accuracy sigma_a (Eq. 12).

        If the adjacency matrix is perfectly hierarchical (i.e., all edges point upward), the optimal beta is infinite, so calling this function will result in a warning and a beta value equal to the self.max_beta instance variable.
        Also, even in cases where the hierarchy is not perfect but is very rigid, the model may overfit, leading to a very large optimal beta value.
        If the optimal beta exceeds self.max_beta, it will be capped at self.max_beta, and a warning will be issued.
        """

        if self._get_proportion_upward_edges() == 0.0:
            if self.warn_beta:
                print(
                    f"Warning: adjacency matrix is perfectly hierarchical, so the optimal beta is infinite. Capping beta at {self.max_beta}."
                )
            return self.max_beta
        elif self.inverse_temp_type == "global":
            return self._get_inverse_temperature_global()
        elif self.inverse_temp_type == "local":
            return self._get_inverse_temperature_local()

    def _get_inverse_temperature_global(self):
        """
        Calculates global inverse temperature parameter beta_L that maximizes the conditional log-likelihood sigma_L (Eq. 13) by performing root-finding on Eq. S39.
        This is the global inverse temperature parameter used for prediction and rank rescaling if the model is fitted with inverse_temp_type="global".
        Root-finding is performed using Brent's method (brentq) over the interval [0.01, 20]. If root-finding fails, beta_L defaults to 20.
        """
        try:
            MLE = brentq(self._eqs39, 0.01, 20, args=(self.ranks, self.A))
            if MLE <= self.max_beta:
                return MLE
        except ValueError:
            pass

        if self.warn_beta:
            print(
                "Warning: Root-finding Eq S39 for beta_L indicates model overfitting because many large beta values are optimal. Capping beta_L at 20. If this is not desired, try increasing the model regularization."
            )
        return self.max_beta

    def _get_inverse_temperature_local(self):
        """
        Calculates local inverse temperature parameter beta_a that minimizes the local accuracy sigma_a (Eq. 12) by minimizing an objective function that approximates the negative of sigma_a.
        This is the local inverse temperature parameter used for prediction and rank rescaling if the model is fitted with inverse_temp_type="local".
        The minimization over a grid of beta values from 0 to 30 using local optimization. If the resulting local minimum is very large (> 20),
        the model is likely overfitting, and beta_a is capped at 20 with a warning.
        """
        bounds = (0, 30)
        with np.errstate(over="ignore"):
            n_grid_points = 5
            beta_grid = np.linspace(
                bounds[0], bounds[1], n_grid_points + 1
            )  # take the best of the local optima
            results = []

            for i in range(n_grid_points):
                res = minimize_scalar(
                    lambda beta: self._eq12_ish(beta, self.ranks, self.A),
                    bounds=(beta_grid[i], beta_grid[i + 1]),
                )
                results.append(res)

            result = min(results, key=lambda r: r.fun)

        beta_a = result.x
        if beta_a > self.max_beta and self.warn_beta:
            print(
                "Warning: Minimizing Eq 12 for local inverse temperature yielded a large local optimum, indicating potential overfitting. Capping beta_a at 20."
            )
        return min(beta_a, self.max_beta)

    def _get_proportion_upward_edges(self):
        """
        Computes the proportion of edges that point upward in the ranking. 
        Used to determine if the adjacency matrix is perfectly hierarchical.
        """
        sorted_indices = np.argsort(self.ranks)[::-1]
        sorted_A = self.A[sorted_indices][:, sorted_indices]

        if scipy.sparse.issparse(sorted_A):
            sorted_A = sorted_A.toarray()
        return np.tril(sorted_A, k=-1).sum() / sorted_A.sum()

    def get_beta(self):
        """
        Get the inverse temperature parameter beta.
        If the inverse_temp_type parameter is "global", returns the global inverse temperature parameter beta_L.
        If the inverse_temp_type parameter is "local", returns the local inverse temperature parameter beta_a.
        """
        if self.is_fitted_beta_ == False:
            self.beta = self._get_inverse_temperature()
            self.is_fitted_beta_ = True
        return self.beta

    def get_rescaled_ranks(self, target_scale):
        """Rescale ranks using target scale and inverse temperature"""
        if self.is_fitted_beta_ == False:
            self.beta = self._get_inverse_temperature()
            self.is_fitted_beta_ = True
        scaling_factor = 1 / (
            np.log(target_scale / (1 - target_scale)) / (2 * self.beta)
        )
        return self.ranks * scaling_factor

    def predict(self, ij_pair):
        """Predict probability that i -> j in a pair [i,j]"""
        if not self.is_fitted_ranks_:
            raise ValueError("Call fit before predicting")
        i = ij_pair[0]
        j = ij_pair[1]
        if not (0 <= i < self.A.shape[0] and 0 <= j < self.A.shape[0]):
            raise ValueError(
                f"Indices {i}, {j} out of bounds for matrix of size {self.A.shape[0]}"
            )
        if self.is_fitted_beta_ == False:
            self.beta = self._get_inverse_temperature()
            self.is_fitted_beta_ = True
        diff = self.ranks[i] - self.ranks[j]
        return 1 / (1 + np.exp(-2 * self.beta * diff))
