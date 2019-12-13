import numpy as np
from scipy.sparse import csr_matrix

import SpringRank

if __name__ == '__main__':
    A = np.zeros((4, 4))
    A[0, 1] = 1

    ranks = SpringRank.get_ranks(A)
    inverse_temperature = SpringRank.get_inverse_temperature(A, ranks)
    scaled_ranks = SpringRank.scale_ranks(ranks, inverse_temperature)

    print("Dense operations:")
    print(scaled_ranks)
    print(SpringRank.get_scaled_ranks(A))

    A_csr = csr_matrix(A)

    sp_ranks = SpringRank.get_ranks(A_csr)
    sp_inverse_temperature = SpringRank.get_inverse_temperature(A_csr.toarray(), sp_ranks)
    sp_scaled_ranks = SpringRank.scale_ranks(sp_ranks, sp_inverse_temperature)

    print("Sparse operations:")
    print(sp_scaled_ranks)
    print(SpringRank.get_scaled_ranks(A_csr.toarray()))
