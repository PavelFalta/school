from time import perf_counter
from functools import *
from collections import deque

# cache = {5:8, 9:2}
#fibonacci 5
# 5 in cache:
#else:
#

@cache
def fibonnaci(n):
    if n == 0:
        return 0
    
    if n == 1:
        return 1
    
    return fibonnaci(n-1) + fibonnaci(n-2)

start = perf_counter()
print(fibonnaci(32))

print(perf_counter()-start)


def brian_kerningan(n):
    print(bin(n))

    count = 0

    while n:
        n = n & (n - 1)
        count += 1

    return count

print(brian_kerningan(255))


class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal(root):
    if root is None:
        return []
    
    return inorder_traversal(root.left) + [root.value] + inorder_traversal(root.right)

def preorder_traversal(root):
    if root is None:
        return []
    
    return [root.value] + preorder_traversal(root.left) + preorder_traversal(root.right)

def postorder_traversal(root):
    if root is None:
        return []
    
    return postorder_traversal(root.left) + postorder_traversal(root.right) + [root.value]

def bfs_traversal(root):
    if root is None:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node.value)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result

def dfs_traversal(root):
    if root is None:
        return []
    
    stack = [root]
    result = []
    
    while stack:
        node = stack.pop()
        result.append(node.value)
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result

# Example usage:
# Constructing a binary tree
#        1
#       / \
#      2   3
#     / \
#    4   5

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

print(inorder_traversal(root))  # Output: [4, 2, 5, 1, 3]
print(preorder_traversal(root))  # Output: [1, 2, 4, 5, 3]
print(bfs_traversal(root))  # Output: [1, 2, 3, 4, 5]
print(dfs_traversal(root))  # Output: [1, 2, 4, 5, 3]
print(postorder_traversal(root))  # Output: [4, 5, 2, 3, 1]