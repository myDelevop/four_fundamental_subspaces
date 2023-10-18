import random
from sympy.matrices import Matrix, eye

if __name__ == '__main__':
    m = random.randint(48, 52)
    n = random.randint(48, 52)
    matrix = Matrix(m, n, lambda i, j: random.randint(-106, 106))

    #    rank_a = matrix.rank()
    #    identity = eye(m)
    # augmented_matrix = matrix.row_join(identity)
    # rref_augmented = augmented_matrix.rref()
    matrix.rref()
