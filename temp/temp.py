import numpy as np
from scipy.sparse import lil_matrix

# 创建一个二维矩阵
matrix = np.array([[1, 2, 0],
                   [0, 3, 4],
                   [5, 0, 6]])

# 创建LIL格式的稀疏矩阵
sparse_matrix = lil_matrix(matrix)
print(sparse_matrix)