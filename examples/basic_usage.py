import numpy as np
from scipy.sparse import csr_matrix
from springrank import SpringRank

# Create sample data
np.random.seed(42); N = 10; A = np.random.binomial(1, 0.3, size=(N, N))

# Initialize and fit model
model = SpringRank(alpha=0.01)
model.fit(A)

# Examine ranks
print("ranks:")
print(model.ranks)

# Predict outcome of a matchup between nodes 1 and 2
print("\nprediction:")
print(model.predict([1,2]))

# Rescale ranks so that a 1-unit difference means a 75% win rate
print("\nrescaled ranks:")
print(model.get_rescaled_ranks(0.75))

# Repeat the process above using a sparse csr matrix
X = csr_matrix(model.A)
model2 = SpringRank(alpha=0.01)
model2.fit(X)

# Examine ranks
print("sparse ranks:")
print(model2.ranks)

# Predict outcome of a matchup between nodes 1 and 2
print("\nsparse prediction:")
print(model2.predict([1,2]))

# Rescale ranks so that a 1-unit difference means a 75% win rate
print("\nsparse rescaled ranks:")
print(model2.get_rescaled_ranks(0.75))