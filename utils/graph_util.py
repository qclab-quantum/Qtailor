import networkx as nx
import matplotlib.pyplot as plt

from networkx.algorithms import isomorphism
from pyvis.network import Network

from utils.points_util import PointsUtil


class GraphUtil():

    #根据邻接矩阵绘制DAG
    @staticmethod
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


    def setOption(self,nt:Network):
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


    def test(self):
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
        self.setOption(nt)
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
        self.setOption(nt)
        nt.from_nx(G1)
        nt.show('nx.html', notebook=False)

    @staticmethod
    def get_new_graph(n:int):
        nodes = list(range(n))
        g = nx.Graph()
        g.add_nodes_from(nodes)
        pair_list = [(nodes[i], nodes[(i + 1) % n]) for i in range(n)]
        g.add_edges_from(pair_list)
        return g

    @staticmethod
    #获取 qiskit 识别的邻接表
    def get_adj(graph:nx.Graph):
        adj = []
        for node, nbrdict in graph.adjacency():
            #print(node, '   ', nbrdict)
            for key in nbrdict:
                adj.append([node, key])
        return adj



if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([0,1, 2, 3, 4])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])

    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
    plt.show()

    adj = GraphUtil.get_adj(G)
    #print(adj)
    #print(PointsUtil.adjacency2matrix(adj))
    print(len(G.edges(1)))
