import numpy as np

# Generate larger matrices for performance testing
size = 100  # You can adjust this size for different performance tests
A = np.random.rand(size, size)  # 100x100 random matrix
B = np.random.rand(size, size)  # 100x100 random matrix

# 1 + 4 + 7

def printmat(A):
    for row in A:
        print(row)

        
def transpose(A):

    return np.array([list(row) for row in list(zip(*A))])

def matmul(A, B):
    assert A.shape[1] == B.shape[0], "Youre stupid"
    result_shape = A.shape[0], B.shape[1]
    result = np.zeros(result_shape)

    B = transpose(B)

    printmat(result)

    for rrow_i, res_row in enumerate(result):
        for ccol_i, _ in enumerate(res_row):
            sum_product = 0
            for A_i, item_A in enumerate(A[rrow_i]):
                sum_product += item_A * B[ccol_i, A_i]
            result[rrow_i, ccol_i] = sum_product

    
    return result



printmat(matmul(A, B))

A = np.array(A)
B = np.array(B)
printmat(A @ B)