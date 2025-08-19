# SpringRank

This is a sparse `numpy` and `scipy` implementation of SpringRank. 

**Paper**: Cate De Bacco, Dan Larremore, and Cris Moore. Science Advances.

**Code**: Dan Larremore, K. Hunter Wapman, Apara Venkateswaran.

# Installation

To develop your own local version of this code, install in editable mode by running the following from the root directory of this repository:
```
pip install -e .
```

To use this code as-is, either clone this repository and install the package locally:
```
pip install .
```

Alternatively, to install SpringRank directly from PyPI, run:
```
pip install springrank
``` 

# Examples

**Get the ranks from a directed adjacency matrix (numpy array)**
```
from springrank import SpringRank
A = np.random.binomial(1, 0.3, size=(10, 10))
# Initialize and fit model
model = SpringRank()
model.fit(A)
# Print the ranks
print(model.ranks)
```

**Compute the inverse temperature parameter (beta) of the ranking and matrix**
```
print(model.get_beta())
```
**Note**: the value of beta depends on which accuracy metric you want to optimize for, specified by the `inverse_temp_type`parameter. 
This can be `"global"` to optimize for the global accuracy (conditional log likelihood) of Eq. 13 or `"local"` to optimize for 
the local accuracy (mean absolute edge prediction error) of Eq. 12. See section **Inverse temperature overfitting** below for more technical details.

**Get the scaled ranks so that a one-rank difference means a 75% win rate**
```
scaled_ranks = model.get_scaled_ranks(0.75)
```

**Include or change regularization alpha (defaults to alpha=0 unless specified)**
```
# Instantiate with regularization 
model = SpringRank(alpha=0.1)
model.fit(A)
print(model.ranks)
# Change the regularization of an existing model
model.alpha = 0.2
model.fit(A)
print(model.ranks)
```

**Make predictions about edge directions**
```
from springrank import SpringRank
A = np.random.binomial(1, 0.3, size=(10, 10))
# Initialize and fit model
model = SpringRank()
model.fit(A)
print("The probability that an undirected edge between 3 and 5 points from 3 to 5 is:\n")
print(model.predict([3,5]))
```

**Inverse temperature overfitting**

If the `inverse_temp_type` parameter is set to `"global"`, the model will solve for the optimal value of the inverse temperature $\hat{\beta_L}$ using Eq. S39.

If the `inverse_temp_type` parameter is set to `"local"`, the model will solve for the optimal value of the inverse temperature $\hat{\beta_a}$ by minimizing a quantity inversely related to $\sigma_a$ derived from Eq. 12:
```math
\hat{\beta_a} = \min_\beta \sum_{i, j}|A_{ij} - \bar{A}_{ij}P_{ij}(\beta)|.
```

If the input adjacency matrix is *perfectly hierarchical*, meaning there are no edges pointing from a lower-ranked node to a higher-ranked node. 
In this case, there must be no reciprocated edges, so $A_{ij} > 0 \to A_{ji} = 0$ for all $i, j$.
Because there is a perfect hierarchy, it must also be the case that for all $i, j$ where $A_{ij} > 0$, $s_i > s_j$.
This means that, in both of the objective functions used to solve for $\beta$, $P_{ij} = 1$ when $A_{ij} > 0$ (or equivalently $s_i > s_j$) and $P_{ij} = 0$ when $A_{ij} = 0$ (or equivalently $s_i < s_j$). Thus the optimization yields $\beta = \infty$.
While this is mathematically correct, huge values of $\beta$ lead to numerical errors in calls to `model.get_rescaled_ranks` and `model.predict`, so we cap $\beta$ at a large value using the `max_beta` model parameter (default `20`). 
If this cap is reached because of perfectly hierarchical input, a warning will be issued if the `warn_beta` flag is set (default `True`).

In some other cases where the input is not perfectly hierarchical but the hierarchy is sufficiently rigid so that there are very few upwards or reciprocated edges, optimal values of $\beta$ (especially $\beta_a$) may become extremely large. In these cases, we again cap $\beta$ at `max_beta` and issue a warning if the `warn_beta` flag is set.