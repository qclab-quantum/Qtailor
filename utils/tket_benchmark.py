import multiprocessing
import os
import re
import time

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

from utils.file.excel_util import ExcelUtil
from utils.file.file_util import FileUtil
from utils.graph_util import GraphUtil as gu
from utils.points_util import PointsUtil as pu
from pytket.utils import Graph
from pytket.passes import PlacementPass, RoutingPass
def draw_graph(coupling_map: List[Union[Tuple[int, int], Tuple[Node, Node]]]):
    coupling_graph = nx.Graph(coupling_map)
    nx.draw(coupling_graph, labels={node: node for node in coupling_graph.nodes()})


'''
compare between qtailor and tket (qtailor/tket)

qft/qft_indep_qiskit_5.qasm:[1, 0, 1, 1, 1, 1, 1, 1, 1, 1] time out

91/93 (qtailor/tket the same below) 
su2/su2random_indep_qiskit_6.qasm:[1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]

48/78
qnn/qnn_indep_qiskit_5.qasm: [1, 0, 1, 1, 1, 1, 1, 0, 1, 0]

29/49
portfolio_vqe/portfoliovqe_indep_qiskit_5.qasm:[1, 1, 1, 1, 1, 1, 1, 0, 1, 1]

69/82
real_amp/realamprandom_indep_qiskit_8.qasm:[1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]

110/120
two_local_ansatz/twolocalrandom_indep_qiskit_10.qasm: [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]

'''

def compile_on_tket(circ):
    # coupling
    # 点阵
    points = [(x, y) for x in range(3) for y in range(3)]

    Nodes = []
    for i in range(len(points)):
        Nodes.append(Node(str(i), points[i]))

    node_coupling = []
    coupling_map = pu.coordinate2adjacent(points)

    for i in range(len(coupling_map)):
        node_coupling.append((Nodes[coupling_map[i][0]], Nodes[coupling_map[i][1]]))

    #show coupling
    # draw_graph(node_coupling)
    # plt.show()

    gird_arch = Architecture(coupling_map)
    mapping_manager = MappingManager(gird_arch)

    # naive_map = {}
    # for i in range(qubits_number):
    #     naive_map[circ.qubits[i]] = Nodes[i]

    #place_with_map(circ, naive_map)

    #PlacementPass(GraphPlacement(gird_arch)).apply(circ)
    PlacementPass(GraphPlacement(gird_arch,maximum_pattern_gates=5000,maximum_matches=5000,maximum_pattern_depth=5000)).apply(circ)
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
    #print(f'compile_on_tket cx={circ.depth_by_type(OpType.CX)}')
   # print(f'compile_on_tket {circ.depth()}')
    return circ.depth()
    # Graph(circ).get_qubit_graph()
    # plt.show()


def compile_on_given_topo(circ,matrix,q_num):
    G = nx.DiGraph()
    # 添加节点
    G.add_nodes_from(range(q_num))

    # 添加边
    for i in range(q_num):
        for j in range(q_num):
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
    for i in range(q_num):
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
    #print(f'compile_on_given_topo circit cx={circ.depth_by_type(OpType.CX)}')

    # print(f'compile_on_given_topo {circ.depth()}')
    # draw_graph(node_coupling)
    # plt.show()
    return circ.depth()

#从字符串中提取出该线路的比特数量 qnn/qnn_indep_qiskit_5.qasm-> 5
def extract_integer(s):
    # Define the regular expression pattern
    pattern = r'[/\_](\d+)\.qasm'

    # Search for the pattern in the string
    match = re.search(pattern, s)

    # If a match is found, extract the integer
    if match:
        return int(match.group(1))
    else:
        return None

def compare_deth(topology,circuit):
    root_dir = FileUtil.get_root_dir()
    sep = os.path.sep
    try:
        q_num = extract_integer(root_dir + os.path.sep + 'benchmark' + circuit)
        matrix = gu.restore_from_1d_array(ast.literal_eval(topology))
        circ = circuit_from_qasm(root_dir + sep + 'benchmark' + sep + circuit)
        original_depth = circ.depth()
        rl_depth = compile_on_given_topo(circ, matrix, q_num)
        tket_depth = compile_on_tket(circ)
        print(f'{circuit}: {rl_depth},{tket_depth},{(rl_depth / tket_depth)}')
        return (rl_depth / tket_depth)

    except Exception as e:
        print(f"Task was interrupted: {e}")

import ast
def benchmark():
# get data
    root_dir = FileUtil.get_root_dir()
    sep = os.path.sep
    sheets,dfs = ExcelUtil.read_by_sheet(root_dir + sep + 'data' + sep+'depth_benchmark_summary.xlsx')
    # pharse data
    for sheet in sheets:
        df = dfs[sheet]
        circuits = df['circuit']
        rl_topology = df['rl_result']
        for i,c  in enumerate(circuits):
            print(c)
            compare_deth(rl_topology[i],c)
            time.sleep(1)

#绘制比较结果，放在 Appendix 中
def plot_result():
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.family'] = ['Arial']
    plt.rcParams['font.size'] = 15
    # 示例数据
    categories = [
                   'Efficient SU2 \n ansatz(6)',
                  'Quantum Neural \nNetwork(10)',
                  'Portfolio Optimization \n with VQE(18)',
                  'Real Amplitudes\n ansatz(8)',
                  'Two Local \n ansatz(10)'
                  ]
    values1 = [91, 168, 328, 69, 110]
    values2 = [93, 253, 539, 82, 120]

    # 设置柱状图的宽度和位置
    bar_width = 0.35
    index = np.arange(len(categories))

    # 创建一个新的图形
    fig, ax = plt.subplots()

    # 绘制两组数据的柱状图
    bars1 = ax.bar(index, values1, bar_width, label='Qtailor', color = '#5370c4',hatch='-', edgecolor='black',zorder=3)
    bars2 = ax.bar(index + bar_width, values2, bar_width, label='Tket', color = '#f16569', hatch='/', edgecolor='black',zorder=3)

    # 在每个柱子的顶部显示具体的值
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom', ha='center')

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom', ha='center')

    # 隐藏上部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(0,600)
    # 添加标签和标题
    #ax.set_xlabel('Circuits')
    ax.set_ylabel('Depth')
    ax.set_title('')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(categories,rotation=30,fontsize=15)
    ax.legend()
    ax.grid(True, which='both', axis='y', linestyle='-', linewidth=1,zorder=0)
    # 显示图形
    plt.tight_layout()

    rootdir = FileUtil.get_root_dir()
    sep = os.path.sep
    path = rootdir + sep + 'data' + sep + 'fig' + sep + 'tket_benchmarkBar.png'

    plt.savefig(path, dpi=300)
    plt.show()


if __name__ == '__main__':

    #benchmark single circuit
    # array =[1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
    # matrix = gu.restore_from_1d_array(array)
    # q_num =8
    # name = r'real_amp/realamprandom_indep_qiskit_8.qasm'
    # root_dir = FileUtil.get_root_dir()

    # circ = circuit_from_qasm(root_dir + '\\benchmark' + os.path.sep + name)
    # rl_depth = compile_on_given_topo(circ, matrix, q_num)
    # tket_depth = compile_on_tket(circ)
    # print(f'{circ}: {rl_depth},{tket_depth},{(rl_depth / tket_depth)}')

    # benchmark all circuits,
    # note that tket do not guarantee that all circuits will compile successfully
    benchmark()

    #plot_result()


