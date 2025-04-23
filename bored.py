# import numpy as np
# from time import perf_counter
# from collections import defaultdict

# # Set the dimensions of the matrices
# # rows_A, cols_A = 3, 2
# # rows_B, cols_B = cols_A, 4  # cols_A must equal rows_B for multiplication

#is this also commited?

# def matmul(A, B):
#     assert A.shape[1] == B.shape[0]
#     sh = A.shape[0], B.shape[1]
#     result = []
#     A = A
#     B = B.T

#     for radek_a in A:
#         for radek_b in B:
#             a = 0
#             for prvek_idx, prvek in enumerate(radek_a):
#                 a+=(prvek * radek_b[prvek_idx])
#             result.append(a)
#     print(np.reshape(result, sh))

# # rows_A, cols_A = 1000, 3
# # rows_B, cols_B = 3, 1000  # cols_A must equal rows_B for multiplication

# # # Create the matrices with random numbers
# # matrix_A = np.random.randint(1, 10, size=(rows_A, cols_A))
# # matrix_B = np.random.randint(1, 10, size=(rows_B, cols_B))

# # print("Matrix A:")
# # print(matrix_A)
# # print("\nMatrix B:")
# # print(matrix_B.T)
# # print(matrix_A @ matrix_B)
# # matmul(matrix_A, matrix_B)

# array = [5, 3, 8, 6]
# indexed_array = sorted(enumerate(array), key=lambda x: x[1])
# sorted_indexes = [index for index, value in indexed_array]
# sorted_values = [value for index, value in indexed_array]

# print("Original array:", array)
# print("Sorted indexes:", sorted_indexes)
# print("Sorted values:", sorted_values)

# from collections import deque

# s = deque()
# # print(s[-1])

# print(1//2)



# def permutations(arr):
#     if len(arr) == 1:
#         return [arr]
#     else:
#         perms = []
#         for i, val in enumerate(arr):
#             for perm in permutations(arr[:i] + arr[i+1:]):
#                 perms.append([val] + perm)
#         return perms
    
# def perm(arr):

#     def backtrack(start):
#         if start == len(arr):
#             res.append(arr[:])
#             return
        
#         for i in range(start, len(arr)):
#             arr[start], arr[i] = arr[i], arr[start]

#             backtrack(start + 1)

#             arr[start], arr[i] = arr[i], arr[start]

#     res = []
#     backtrack(0)
#     return res


# # start = perf_counter()
# # p1 = permutations([1, 2, 3,4,5,6,7,8,9])
# # print(perf_counter() - start)


# # start = perf_counter()
# # p2 = perm([1, 2, 3,4,5,6,7,8,9])
# # print(perf_counter() - start)


# def ffib(n):
#     def fib_helper(k):
#         if k == 0:
#             return (0,1)
        
#         a, b = fib_helper(k // 2)
#         c = a * ((b << 1) - a)
#         d = a ** 2 + b ** 2

#         if k % 2 == 0:
#             return (c, d)
#         else:
#             return (d, c + d)
        
#     return fib_helper(n)[0]
    
# def fib(n):
#     if n <= 0:
#         return 0
    
#     elif n == 1:
#         return 1
    
#     a, b = 0, 1 
#     for _ in range(2, n + 1):
#         a, b = b, a + b
    
#     return b

# def sfib(n):
#     sqrt5 = 5 ** 0.5

#     psi = (1 + sqrt5) / 2
#     phi = (1 - sqrt5) / 2
    
#     fib = (psi ** n - phi ** n) / sqrt5
#     return round(fib)

# # print(ffib(858100))
# # print(sfib(1500))

# start = perf_counter()
# ffib(1200)
# print(perf_counter() - start)

# start = perf_counter()
# sfib(1200)
# print(perf_counter() - start)


# start = perf_counter()
# print(fib(9))
# print(perf_counter() - start)

# def myperm(arr):

#     def backtrack(start):
#         if start == len(arr):
#             res.append(arr[:])
#             return

#         for i in range(start, len(arr)):

#             arr[start], arr[i] = arr[i], arr[start]
#             backtrack(start+1)
#             arr[start], arr[i] = arr[i], arr[start]


#     res = []
#     backtrack(0)
#     return res


# print(myperm([1,2,3]))


# def bfs_arr(graph, start, finish):

#     adjancency_dict = defaultdict(list)

#     for u,v in graph:
#         adjancency_dict[u].append(v)
#         adjancency_dict[v].append(u)

    
#     queue = deque([(start, [])])
#     visited = set([start])

#     while queue:
#         current, path = queue.popleft()

#         if current == finish:
#             return path + [current]

#         for neighbor in adjancency_dict[current]:
#             if neighbor not in visited:
#                 queue.append((neighbor, path+[neighbor]))
#                 visited.add(neighbor)


    
# print(bfs_arr([[0,1],[1,5],[1,9],[9,33],[39,6],[33,91],[91,39],[6,3]], 0, 3))


# def mat_rotate_90(matrix):
#     return list(zip(*matrix[::-1]))



# print(mat_rotate_90([[1,2,3],[4,5,6],[7,8,9]]))



# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'D': ['F'],
#     'E': ['G'],
#     'F': ['H'],
#     'G': ['H', 'I'],
#     'H': ['J'],
#     'I': ['J'],
#     'J': []
# }


# def topological_sorting(graph):
#     degrees = {i:0 for i, _ in graph.items()}
    

#     for u in graph:
#         for v in graph[u]:
#             degrees[v] += 1
    
#     print(degrees)
    
#     queue = deque([u for u,v in degrees.items() if v == 0])

#     topo = []

#     while queue:
#         current = queue.popleft()
#         topo.append(current)

#         for neighbor in graph[current]:
#             degrees[neighbor] -= 1
#             if not degrees[neighbor]:
#                 queue.append(neighbor)

#     return topo

# def shortest_path_dag(graph, start):
#     topo = topological_sorting(graph=graph)


#     distances = {i:float("inf") for i, _ in graph.items()}
#     distances[start] = 0

#     for node in topo:
#         if not distances[node] == float("inf"):
#             for neighbor in graph[node]:
#                 if distances[node] + 1 < distances[neighbor]:
#                     distances[neighbor] = distances[node] + 1

#     return distances

# print(topological_sorting(graph=graph))
# print(shortest_path_dag(graph=graph, start="A"))


def spoj_slova(lst):
    return "".join(lst)

# print(spoj_slova(["ahoj", "debilku"])) #ahojdebilku

s = ""


cisla = ""

# print("ahua" in cisla)

a = set()

a.add("asdasd")
a.remove("asdasd")


class LinkedList:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next

def reverse(head):
    if not head or not head.next:
        return head

    rev_head = reverse(head.next)

    head.next.next = head
    head.next = None

    return rev_head

def walk(head):
    if head:
        print(head.val)
        walk(head.next)


#generate some linked lists
head = LinkedList(1)

current = head

for i in range(2, 10):
    current.next = LinkedList(i)
    current = current.next

# walk(reverse(head))


# make the BST data structure and generate example BST
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def insert(root, val):
    if not root:
        return Node(val)

    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)

    return root

#generate the BST
root = Node(5)
insert(root, 3)
insert(root, 7)
insert(root, 2)
insert(root, 4)
insert(root, 6)
insert(root, 8)

def inorder(root):
    if not root:
        return []
    
    return inorder(root.left) + [root.val] + inorder(root.right)

print(inorder(root))

def find(root, val):
    if not root:
        return False

    if root.val == val:
        return True

    if val < root.val:
        return find(root.left, val)
    else:
        return find(root.right, val)
    
print(find(root, 8))

rules = {"A": "B-A-B",
         "B": "A+B+A"}

word = ["A"]

epochs = 2
for i in range(epochs):
    for idx, letter in enumerate(word):
        if letter in rules:
            for rule in rules[letter]:
                word.insert(idx+1, rule)
        
        print(word)
        input()
    
    print(word)

