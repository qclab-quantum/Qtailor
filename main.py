array = [0, 1, 2, 3, 4,5]  # 给定的有序数组

pair_list = [(array[i], array[(i+1)%len(array)]) for i in range(len(array))]
print(pair_list)