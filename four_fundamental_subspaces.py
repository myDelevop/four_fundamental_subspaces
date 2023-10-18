from sympy.matrices import Matrix


if __name__ == '__main__':

    # Define a static Matrix A
    A = Matrix([
        [1, 2, 2, 3],
        [2, 4, 1, 3],
        [3, 6, 1, 4]
    ])

    # m is the number of rows of the matrix and n is the number of columns
    (m, n) = A.shape
