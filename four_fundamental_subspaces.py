import numpy as np
from sympy.matrices import Matrix, eye


def compute_subspaces(matrix):
    # m is the number of rows of the matrix and n is the number of columns
    (m, n) = matrix.shape

    # number of pivotal elements in Matrix A
    rank_a = matrix.rank()

    # Create an identity matrix of size m (num of rows)
    identity = eye(m)

    # Create the augmented matrix [A|I] with identity.
    augmented_matrix = matrix.row_join(identity)

    # Computation of row reduced Echelon form
    rref_augmented = augmented_matrix.rref()

    # Separate A and I from [A|I]
    a_rref = rref_augmented[0][:, :m + 1]
    id_rref = rref_augmented[0][:, m + 1:]
    pivots_tmp = np.array(rref_augmented[1])
    """
    Be careful, since it is the augmented matrix A, the indexes con return more
    elements for the pivotal element (because there is a reduction of the Identity
    Matrix. That's why we take the elements :rank_A (from 0 to rank_A)
    """
    # The indices where pivots appear in the matrix A (we stop to rank of A because there are more)
    pivot_idx = np.array(rref_augmented[1][:rank_a])

    matrix_computation = {
        "matrix": matrix,
        "shape": (m, n),
        "num_rows": m,
        "num_cols": n,
        "computation_time_rank": 0,
        "computation_time_rref": 0,
        "computation_time_tot": 0,
        "rank": rank_a,
        "identity": identity,
        "augmented_matrix": augmented_matrix,
        "rref_augmented": rref_augmented,
        "a_rref": a_rref,
        "id_rref": id_rref,
        "pivot_idx": pivot_idx,
        "pivots_tmp": pivots_tmp,
        "four_subspaces": {
            "Range_A": {"span": []},
            "Range_AT": {"span": []},
            "NULL_A": {"span": []},
            "NULL_AT": {"span": []},
        }
    }

    ''''
     FOUR FUNDAMENTAL SUBSPACES COMPUTATION
    '''

    for i in range(0, rank_a):
        idx = pivot_idx[i]
        col_idx = matrix.col(idx)
        row_idx = rref_augmented[0].row(i)[:n]  # is R but in sympy library
        matrix_computation["four_subspaces"]["Range_A"]["span"].append(col_idx)
        matrix_computation["four_subspaces"]["Range_AT"]["span"].append(row_idx)
        matrix_computation["four_subspaces"]["NULL_A"]["span"] = matrix.nullspace()

    if rank_a <= m:
        np_matrix = np.array(id_rref)  # np_matrix is the numpy matrix of sympy
        matrix_computation["four_subspaces"]["NULL_AT"]["span"] = np_matrix[-(m - rank_a):].tolist()
    else:
        matrix_computation["four_subspaces"]["NULL_AT"]["span"] = [0] * m


if __name__ == '__main__':

    # Define a static Matrix A
    A = Matrix([
        [1, 2, 2, 3],
        [2, 4, 1, 3],
        [3, 6, 1, 4]
    ])

    compute_subspaces(matrix=A)




