import numpy as np

# 创建一个多维 NumPy 数组
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 使用 array2string 方法并指定 separator 参数
arr_str = np.array2string(arr, separator=', ')

# 打印结果
print(arr_str)