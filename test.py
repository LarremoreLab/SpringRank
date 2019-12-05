import numpy as np
import SpringRank

if __name__ == '__main__':
    A = np.zeros((4, 4))
    A[0, 1] = 1

    ranks = SpringRank.get_ranks(A)
    inverse_temperature = SpringRank.get_inverse_temperature(A, ranks)
    scaled_ranks = SpringRank.scale_ranks(ranks, inverse_temperature)

    print(scaled_ranks)
    print(SpringRank.get_scaled_ranks(A))
