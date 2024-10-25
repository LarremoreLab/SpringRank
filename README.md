# SpringRank

This is a sparse `numpy` and `scipy` implementation of SpringRank. 

**Paper**: Cate De Bacco, Dan Larremore, and Cris Moore. Science Advances.

**Code**: Dan Larremore, K. Hunter Wapman, Apara Venkateswaran.

# Installation

```
pip install -r requirements.txt
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
print("The probability that an undirected edge beween 3 and 5 points from 3 to 5 is:\n")
print(model.predict([3,5]))
```