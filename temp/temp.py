import time

matrices = set()

#生成矩阵并存储到集合中
for i in range(500000):
    matrix = tuple(range(i, i+100))
    matrices.add(frozenset(matrix))

print('done')
time.sleep(100)