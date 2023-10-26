

def adjacency_matrix(adj_list):
    max_index = max(max(pair) for pair in adj_list)
    matrix = [[0] * (max_index + 1) for _ in range(max_index + 1)]

    for pair in adj_list:
        matrix[pair[0]][pair[1]] = 1
        matrix[pair[1]][pair[0]] = 1

    return matrix

adj_list = [[0,1], [1,2], [2,3]]
matrix = adjacency_matrix(adj_list)

for row in matrix:
    print(row)

