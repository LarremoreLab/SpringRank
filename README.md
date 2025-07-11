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
Note: the value of beta depends on which accuracy metric you want to optimize for, specified by the `inverse_temp_type`parameter. 
This can be `"global"` for the global accuracy (conditional log likelihood) of Eq. 13 or `"local"` for 
the local accuracy (mean absolute edge prediction error) of Eq. 12.

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