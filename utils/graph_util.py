import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

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

    node_labels = {0: 'Q0', 1: 'B', 2: 'C', 3: 'D'}

    for i in range(9):
        node_labels[i] = 'Q' + str(i)
    # 绘制有向无环图
    pos = nx.spring_layout(G)
    nx.draw(G, pos,with_labels=False, node_color='lightblue', node_size=500, arrows=True)

    # 设置节点标签
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.show()
# 邻接表

# points = [(0,0),(0,1),(0,2),
#           (1,0),(1,1),(1,2)
#           ,(2,0),(2,1),(2,2)]
# adj = coordinate2adjacent(points)

# adj = [[0, 1], [0, 3], [1, 0], [1, 2], [1, 4], [2, 1],
#        [2, 5], [3, 0], [3, 4], [3, 6], [4, 1], [4, 3], [4, 5],
#        [4, 7], [5, 2], [5, 4], [5, 8], [6, 3], [6, 7], [7, 4],
#        [7, 6], [7, 8], [8, 5], [8, 7]]

# adj = [
#     [0,1],[1,0]
# ]

def test():
    # 创建一个空的无向图
    G = nx.Graph()

    # 添加节点
    num_nodes = 5  # 这里可以根据你的需求指定节点数量
    nodes = range(num_nodes)
    G.add_nodes_from(nodes)

    # 遍历所有节点并添加边
    for node in nodes:
        neighbors = [n for n in nodes if n != node]
        G.add_edges_from([(node, n) for n in neighbors[:2]])

    # 输出图的节点个数和边数
    # print("节点数量:", G.number_of_nodes())
    # print("边数量:", G.number_of_edges())
    # nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
    # plt.show()
    nt = Network('500px', '500px')
    nt.from_nx(G)
    nt.show('nx.html')
def demo():
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
    plt.show()

def setOption(nt:Network):
    nt.set_options("""
        var options = {
        "nodes": {
        "borderWidth": null,
        "borderWidthSelected": null,
        "opacity": null,
        "size": null
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "selfReferenceSize": null,
        "selfReference": {
          "angle": 0.7853981633974483
        },
        "smooth": false
      },
      "physics": {
        "enabled": false,
        "minVelocity": 0.75
      }
    }
        """)
def test1():
    from pyvis.network import Network
    import networkx as nx
    nx_graph = nx.cycle_graph(10)
    nx_graph.nodes[1]['title'] = 'Number 1'
    nx_graph.nodes[1]['group'] = 1
    nx_graph.nodes[3]['title'] = 'I belong to a different group!'
    nx_graph.nodes[3]['group'] = 10
    nx_graph.add_node(20, size=20, title='couple', group=2)
    nx_graph.add_node(21, size=15, title='couple', group=2)
    nx_graph.add_edge(20, 21, weight=5)
    nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
    # populates the nodes and edges data structures
    nt = Network('1000px', '1000px')
    setOption(nt)
    nt.from_nx(nx_graph)
  #  nt.show_buttons()

    nt.show('nx.html',notebook=False)
if __name__ == '__main__':
    test1()