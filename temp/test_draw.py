import networkx as nx
import matplotlib.pyplot as plt


# 根据点的二维坐标生成邻接表
def coordinate2adjacent(points):
    import math
    point_dict = {i: points[i] for i in range(len(points))}
    adjacency_dict = {}

    for i, point in point_dict.items():
        adjacent_points = []
        for j, other_point in point_dict.items():
            if i != j:
                if math.sqrt((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2) == 1:
                    adjacent_points.append(j)
        adjacency_dict[i] = adjacent_points

    #transform adjacency_dict to qiskit format
    res = []
    for k in adjacency_dict:
        v = adjacency_dict.get(k)
        for node in v:
            res.append([k,node])
    # return adjacency_dict
    return res


#根据邻接表生成邻接矩阵
def adjacency_matrix(adj_list):
    max_index = max(max(pair) for pair in adj_list)
    matrix = [[0] * (max_index + 1) for _ in range(max_index + 1)]

    for pair in adj_list:
        matrix[pair[0]][pair[1]] = 1
        matrix[pair[1]][pair[0]] = 1

    return matrix
#根据邻接矩阵绘制DAG
def draw_dag(adj_matrix):
    # 创建有向无环图对象
    G = nx.DiGraph()

    # 添加节点
    num_nodes = len(adj_matrix)
    G.add_nodes_from(range(num_nodes))

    # 添加边
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)

    # 绘制有向无环图
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, arrows=True)
    plt.show()
# 邻接表
# adj = [
#     [0, 1],
#     [1, 2],
#     [2, 3]
# ]
points = [(0,0),(0,1),(0,2),
          (1,0),(1,1),(1,2)
          ,(2,0),(2,1),(2,2)]
adj = coordinate2adjacent(points)
print(adjacency_matrix(adj))
# 示例邻接矩阵
adj_matrix = adjacency_matrix(adj)
# adj_matrix = [[0, 1, 1, 0],
#               [0, 0, 1, 1],
#               [0, 0, 0, 0],
#               [0, 0, 1, 0]]



# 绘制有向无环图
draw_dag(adj_matrix)