# SpringRank

This is a sparse `numpy` and `scipy` implementation of SpringRank. 

**Paper**: Cate De Bacco, Dan Larremore, and Cris Moore. Science Advances.

**Code**: Dan Larremore, K. Hunter Wapman, Apara Venkateswaran.

# Examples

**Get the ranks from a directed adjacency matrix (numpy array)**
```import SpringRank as sr
A = np.zeros((4, 4))
A[0, 1] = 1
ranks = sr.get_ranks(A)
```

**Compute the inverse temperature of the ranking and matrix**
```
ranks = sr.get_ranks(A)
inverse_temperature = sr.get_inverse_temperature(A, ranks)
```

**Get the scaled ranks so that a one-rank difference means a 75% win rate**
```
scaled_ranks = sr.get_scaled_ranks(A,scale=0.75)
```