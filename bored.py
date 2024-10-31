import numpy as np

# Set the dimensions of the matrices
# rows_A, cols_A = 3, 2
# rows_B, cols_B = cols_A, 4  # cols_A must equal rows_B for multiplication

def matmul(A, B):
    assert A.shape[1] == B.shape[0]
    sh = A.shape[0], B.shape[1]
    result = []
    A = A
    B = B.T

    for radek_a in A:
        for radek_b in B:
            a = 0
            for prvek_idx, prvek in enumerate(radek_a):
                a+=(prvek * radek_b[prvek_idx])
            result.append(a)
    print(np.reshape(result, sh))

# rows_A, cols_A = 1000, 3
# rows_B, cols_B = 3, 1000  # cols_A must equal rows_B for multiplication

# # Create the matrices with random numbers
# matrix_A = np.random.randint(1, 10, size=(rows_A, cols_A))
# matrix_B = np.random.randint(1, 10, size=(rows_B, cols_B))

# print("Matrix A:")
# print(matrix_A)
# print("\nMatrix B:")
# print(matrix_B.T)
# print(matrix_A @ matrix_B)
# matmul(matrix_A, matrix_B)

array = [5, 3, 8, 6]
indexed_array = sorted(enumerate(array), key=lambda x: x[1])
sorted_indexes = [index for index, value in indexed_array]
sorted_values = [value for index, value in indexed_array]

print("Original array:", array)
print("Sorted indexes:", sorted_indexes)
print("Sorted values:", sorted_values)

from collections import deque

s = deque()
# print(s[-1])

print(1//2)