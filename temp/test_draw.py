import networkx as nx
import matplotlib.pyplot as plt

def adjacency_matrix(adj_list):
    max_index = max(max(pair) for pair in adj_list)
    matrix = [[0] * (max_index + 1) for _ in range(max_index + 1)]

    for pair in adj_list:
        matrix[pair[0]][pair[1]] = 1
        matrix[pair[1]][pair[0]] = 1

    return matrix

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
adj = [
    [0, 1],
    [1, 2],
    [2, 3]
]
# 示例邻接矩阵
adj_matrix = adjacency_matrix(adj)
# adj_matrix = [[0, 1, 1, 0],
#               [0, 0, 1, 1],
#               [0, 0, 0, 0],
#               [0, 0, 1, 0]]



# 绘制有向无环图
draw_dag(adj_matrix)