import random
from sympy.matrices import Matrix, eye


if __name__ == '__main__':

    m = random.randint(58, 62)
    n = random.randint(58, 62)
    A = Matrix(m, n, lambda i, j: random.randint(-14, 14))

    A.rref()
