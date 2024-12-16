import os

from matplotlib import pyplot as plt
from pytket import OpType
from pytket._tket.passes import DecomposeSwapsToCXs
from pytket.architecture import Architecture
from pytket.circuit import Node
import networkx as nx
from typing import List, Union, Tuple

from pytket.circuit.display import render_circuit_as_html
from pytket.qasm import circuit_from_qasm, circuit_to_qasm_str
from pytket.mapping import MappingManager
from pytket.mapping import LexiLabellingMethod, LexiRouteRoutingMethod,AASRouteRoutingMethod, AASLabellingMethod
from pytket.placement import place_with_map, GraphPlacement
from utils.file.file_util import FileUtil
from utils.graph_util import GraphUtil as gu
from utils.points_util import PointsUtil as pu
from pytket.utils import Graph
from pytket.passes import PlacementPass, RoutingPass
def draw_graph(coupling_map: List[Union[Tuple[int, int], Tuple[Node, Node]]]):
    coupling_graph = nx.Graph(coupling_map)
    nx.draw(coupling_graph, labels={node: node for node in coupling_graph.nodes()})

'''
real_amp:[1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]

'''

array =[1, 1, 1, 1, 1, 1, 1, 0, 1, 1]





matrix = gu.restore_from_1d_array(array)
qubits_number = 5
name = r'portfolio_vqe/portfoliovqe_indep_qiskit_5.qasm'
root_dir = FileUtil.get_root_dir()

def compile_on_tket():
    circ = circuit_from_qasm(root_dir + '\\benchmark' + os.path.sep + name)
    # coupling
    # 点阵
    points = [(x, y) for x in range(4) for y in range(4)]

    Nodes = []
    for i in range(len(points)):
        Nodes.append(Node(str(i), points[i]))

    node_coupling = []
    coupling_map = pu.coordinate2adjacent(points)

    for i in range(len(coupling_map)):
        node_coupling.append((Nodes[coupling_map[i][0]], Nodes[coupling_map[i][1]]))
    #
    draw_graph(node_coupling)
    plt.show()

    gird_arch = Architecture(coupling_map)
    mapping_manager = MappingManager(gird_arch)

    # naive_map = {}
    # for i in range(qubits_number):
    #     naive_map[circ.qubits[i]] = Nodes[i]

    #place_with_map(circ, naive_map)

    PlacementPass(GraphPlacement(gird_arch,maximum_pattern_gates=3000,maximum_matches=3000,maximum_pattern_depth=3000)).apply(circ)
    mapping_manager.route_circuit(circ, [
        AASRouteRoutingMethod(1),
        LexiLabellingMethod(),
        LexiRouteRoutingMethod(),
        AASLabellingMethod(),
        ])
    # DecomposeSwapsToCXs(gird_arch, True).apply(circ)
    # from pytket.extensions.qiskit import tk_to_qiskit
    # circ = tk_to_qiskit(circ)
    #circ.draw('mpl').show()
    print(f'compile_on_tket cx={circ.depth_by_type(OpType.CX)}')
    print(f'compile_on_tket {circ.depth()}')
    # Graph(circ).get_qubit_graph()
    # plt.show()


def compile_on_given_topo():
    circ = circuit_from_qasm(root_dir + '\\benchmark' + os.path.sep + name)
    G = nx.DiGraph()
    # 添加节点
    num_nodes = len(matrix)
    G.add_nodes_from(range(num_nodes))

    # 添加边
    for i in range(num_nodes):
        for j in range(num_nodes):
            if matrix[i][j] == 1:
                G.add_edge(i, j)

    coupling_map = gu.get_adj_list(G)
    #print(coupling_map)
    Nodes = []
    for i in range(len(coupling_map)):
        Nodes.append(Node(str(i), i))

    node_coupling = []

    for i in range(len(coupling_map)):
        node_coupling.append((Nodes[coupling_map[i][0]], Nodes[coupling_map[i][1]]))

    # draw_graph(node_coupling)
    arch = Architecture(coupling_map)
    mapping_manager = MappingManager(arch)

    naive_map = {}
    for i in range(qubits_number):
        naive_map[circ.qubits[i]] = Nodes[i]

    place_with_map(circ, naive_map)
    # html =  render_circuit_as_html(circ)
    # print(html)

    #PlacementPass(GraphPlacement(arch,maximum_pattern_gates=9999,maximum_matches=9999)).apply(circ)
    mapping_manager.route_circuit(circ, [
        AASRouteRoutingMethod(1),
        LexiLabellingMethod(),
        LexiRouteRoutingMethod(),
        AASLabellingMethod(), ])

    from pytket.extensions.qiskit import tk_to_qiskit
    #circ = tk_to_qiskit(circ)
    #circ.draw('mpl').show()

    # DecomposeSwapsToCXs(gird_arch, True).apply(circ)
    print(f'compile_on_given_topo circit cx={circ.depth_by_type(OpType.CX)}')
    print(f'compile_on_given_topo {circ.depth()}')
    draw_graph(node_coupling)
    plt.show()


if __name__ == '__main__':
    circ = circuit_from_qasm(root_dir + '\\benchmark' + os.path.sep + name)
    print(f'original circit depth {circ.depth()}')
    compile_on_given_topo()
    compile_on_tket()



