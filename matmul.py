import numpy as np

A = np.array(
    [[1,2,3],
     [4,5,6],
     [7,8,9]])

B = np.array(
    [[1],
     [4],
     [7]])

def printmat(A):
    for row in A:
        print(row)

        
def transpose(A):

    return np.array([list(row) for row in list(zip(*A))])

def matmul(A, B):
    A = transpose(A)

    result_shape = A.shape[0], B.shape[1]
    result = np.zeros(result_shape)
    printmat(result)

    for row_B in B:
        for row_A in A:

            for item_B in row_B:
                for item_A in row_A:
                    ...




printmat(transpose(B))
matmul(A, B)

A = np.array(A)
B = np.array(B)
print(A @ B)