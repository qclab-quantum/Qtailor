import numpy as np

def positional_encoding_2D(x, y, d_model):
    pe = np.zeros((d_model,))
    for i in range(d_model):
        if i % 2 == 0:
            pe[i] = np.sin(x / (10000 ** (2 * i / d_model)))
        else:
            pe[i] = np.cos(y / (10000 ** ((2 * i - 1) / d_model)))
    return pe

# 创建一个 10x10 的二维矩阵
matrix = np.zeros((10, 10, 16))  # 调整矩阵的形状为 10x10x16

# 对矩阵的每个位置进行编码
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        matrix[i, j] = positional_encoding_2D(i, j, d_model=16)  # 假设位置编码的维度为16

print(matrix[0][0])
print(matrix[0][1])
print(matrix[0][2])