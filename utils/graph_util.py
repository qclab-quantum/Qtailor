import math
import multiprocessing
import time
import random

import karateclub
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from pyvis.network import Network

from utils.circuit_util import CircutUtil
from utils.points_util import PointsUtil

from qiskit_aer import AerSimulator

simulator = AerSimulator()
from multiprocessing import Pool

# 设置 graph 样式
def setOption( nt: Network):
    nt.set_options("""
           var options = {
         "edges": {
           "smooth": false
         },
         "physics": {
           "enabled": false,
           "minVelocity": 0.75
         }
       }
           """)


class GraphUtil():


    def __init__(self, ):
        self.pool = Pool(10)
    @staticmethod
    def draw_adj_list(adj_list:list,node_number):
        g = nx.Graph()
        g.add_nodes_from(list(range(node_number)))
        g.add_edges_from(adj_list)
        nx.draw(g, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
        plt.show()

    #根据邻接矩阵绘制DAG
    @staticmethod
    def draw_adj_matrix(adj_matrix,is_draw_nt=False):
        # 创建图对象
        G = nx.DiGraph()

        # 添加节点
        num_nodes = len(adj_matrix)
        G.add_nodes_from(range(num_nodes))

        # 添加边
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i][j] == 1:
                    G.add_edge(i, j)

        node_labels = {0: 'Q0',}

        for i in range(num_nodes):
            node_labels[i] = 'Q' + str(i)
        # 绘制有向无环图
        pos = nx.spring_layout(G)
        nx.draw(G, pos,with_labels=False, node_color='lightblue', node_size=500, arrows=True)

        # 设置节点标签
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        plt.show()
        if is_draw_nt:
            nt = Network('1000px', '1000px')
            setOption(nt)
            nt.from_nx(G)
            for i in range(len(nt.nodes)):
                nt.nodes[i]['label'] = 'Q' + str(nt.nodes[i]['id'])
                #print(nt.nodes[i])
            nt.show('nx.html', notebook=False)


    def demo(self):
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


    def generate_html(self):

        #add_edge('0 2 0 3 1 2 1 3 2 4 3 5 4 6 5 6 4 7 5 7 6 7 ')
        G1 = nx.Graph()
        G1.add_nodes_from(range(3))
        G1.add_edge(0, 2)
        G1.add_edge(0, 1)
        #print(isomorphism.GraphMatcher(G1, G2).is_isomorphic())
        nt = Network('1000px', '1000px')
        setOption(nt)
        nt.from_nx(G1)
        nt.show('nx.html', notebook=False)

    @staticmethod
    def get_new_graph(n:int):
        nodes = list(range(n))
        g = nx.Graph()
        g.add_nodes_from(nodes)
        pair_list = [(nodes[i], nodes[(i + 1) % n]) for i in range(n)]
        g.add_edges_from(pair_list)
        #去掉一条线，使圆变成线
        g.remove_edge(0,n-1)
        return g

    @staticmethod
    #为所有的点之间增加边，将图变为全连接
    def set_full_conn(graph:nx.Graph):
        n = len(graph.nodes)
        for i in range(n):
            for j in range(n):
                if i != j and not graph.has_edge(i,j):
                    graph.add_edge(i,j)

    @staticmethod
    #获取 qiskit 识别的邻接表
    def get_adj_list(graph:nx.Graph):
        adj = []
        for node, nbrdict in graph.adjacency():
            #print(node, '   ', nbrdict)
            for key in nbrdict:
                adj.append([node, key])
        return adj

    @staticmethod
    # 获取 graph 的邻接矩阵
    def get_adj_matrix(graph:nx.Graph):
        return np.array(nx.adjacency_matrix(graph).todense())

    @staticmethod
    def add_edge_to_matrix(matrix, i, j):
        if i == j: return

        matrix[i][j] = 1
        matrix[j][i] = 1

    @staticmethod
    def del_edge_from_matrix(matrix, i, j):
        matrix[i][j] = 0
        matrix[j][i] = 0

    @staticmethod
    def test_adj_matrix(adj_matrix, qasm):
        circuit = CircutUtil.get_from_qasm(qasm)
        G = nx.DiGraph()
        # 添加节点
        num_nodes = len(adj_matrix)
        G.add_nodes_from(range(num_nodes))

        # 添加边
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i][j] == 1:
                    G.add_edge(i, j)

        adj_list = GraphUtil.get_adj_list(G)
        layout = list(range(len(circuit.qubits)))
        avr_rl = 0
        avr_rl_mix = 0
        result = []
        repeat = 10
        for i in range(repeat):
            try:
                ct1 = transpile(circuits=circuit, coupling_map=adj_list, initial_layout=layout,  optimization_level=3, backend=simulator)
                ct2 = transpile(circuits=circuit, coupling_map=adj_list, optimization_level=3, backend=simulator)
                d1 = ct1.decompose().depth()
                d2 = ct2.decompose().depth()
                result.append([d1,d2])
                avr_rl += d1
                avr_rl_mix += d2
            except Exception as e:
                print(e)
                result.append([-1,-1])

            # print(ct.layout.initial_layout)
        avr_rl /= repeat
        avr_rl_mix /= repeat
        result.append([avr_rl,avr_rl_mix])
        return result

    @staticmethod
    #将矩阵左下三角拉伸为一维矩阵
    def lower_triangle_to_1d_array( matrix):
        rows = len(matrix)
        cols = rows
        result = []

        for i in range(rows):
            for j in range(cols):
                if j < i:  # 仅处理左下三角部分
                    result.append(matrix[i][j])
        return result
    @staticmethod
    #n = ( -1 + sqrt(1 + 8S) ) / 2
    #将输出的一维 obs 恢复成 二维矩阵
    def restore_from_1d_array(array):
        length = int((math.sqrt(8 * len(array) +1) -1 ) / 2) + 1
        #length*length的矩阵
        matrix = [[0] * length for _ in range(length)]

        index = 0
        for i in range(1,length):
            for j in range(0,i):
                matrix[i][j] =array[index]
                matrix[j][i] = array[index]
                index += 1

        return matrix


def test_adj_list(adj):
    g = nx.Graph()
    g.add_nodes_from([0,1,2,3,4])
    #[[0, 1], [0, 4], [0, 2], [1, 0], [1, 2], [2, 1], [2, 3], [2, 0], [3, 2], [3, 4], [4, 3], [4, 0]]
    g.add_edges_from(adj)
    nx.draw(g, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
    plt.show()

    circuit = QuantumCircuit(5)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    circuit.cx(0, 4)
    print(CircutUtil.get_circuit_score1(circuit,adj=adj))



#测试 Qiskit 在田字格上的编译结果
def test_tian():
    points = []
    for i in range(10):
        for j in range(10):
            points.append((i, j))

    adj = PointsUtil.coordinate2adjacent(points)
    PointsUtil.plot_points(points)

    @staticmethod
    #调用 draw_adj_matrix 实现绘制拓扑图
    def draw_1d_array(array):
        graph = GraphUtil.restore_from_1d_array(array)
        GraphUtil.draw_adj_matrix(graph)
        print(GraphUtil.restore_from_1d_array(array))

if __name__ == '__main__':
    from karateclub import EgoNetSplitter
    wc =karateclub.WaveletCharacteristic(eval_points = 3)
    g = nx.Graph()
    g.add_nodes_from(range(3))
    g.add_edge(0, 2)
    g.add_edge(0, 1)
    gc =karateclub.Graph2Vec()

    #wc.fit(graphs=[g])
    gc.fit(graphs=[g])
    print(gc.get_embedding())

