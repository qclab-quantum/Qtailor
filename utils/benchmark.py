import time
import traceback
from collections import OrderedDict

import networkx as nx
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from fidelity import circuit_fidelity_benchmark
from utils.circuit_util import CircutUtil as cu, CircutUtil
from utils.file.csv_util import CSVUtil
from utils.graph_util import GraphUtil as gu, GraphUtil
from utils.points_util import PointsUtil as pu
from qiskit.converters import circuit_to_dag, dag_to_circuit
#15*15 的点阵
points  = [(x, y) for x in range(15) for y in range(15)]

# points = [(0, 0), (0, 1), (0, 2), (0, 3),
#           (1, 0), (1, 1), (1, 2), (1, 3),
#           (2, 0), (2, 1), (2, 2), (2, 3),
#           (3, 0), (3, 1), (3, 2), (3, 3)]
class Benchmark():

    def __init__(self,qasm ):
        self.qasm = qasm

    @staticmethod
    def depth_benchmark(file_path,matrix:np.ndarray,qasm:str,is_draw=False):
        # b_1[
        # [rl * n]
        # [rl_qiskit * n]
        # ]
        b_1 = gu.test_adj_matrix(matrix, qasm)
        rl = b_1[-1][0]
        mix = b_1[-1][1]
        #get Qiskit result
        b_2 = Benchmark.get_qiskit_depth(qasm)
        qiskit = b_2[-1]
        print('rl = %r,qiskit= %r, mix = %r '%(rl,qiskit,mix))

        #write to csv file
        data = []
        for i in range(len(b_1)):
            data.append(['','',b_1[i][0],b_2[i],b_1[i][1]])
        CSVUtil.append_data(file_path, data)
        if is_draw:
            gu.draw_adj_matrix(matrix,is_draw_nt=True)
            pu.plot_points(points)
        return rl,qiskit,mix

    @staticmethod
    def get_qiskit_depth(qasm:str):
        result = []
        repeat = 10
        adj_list = pu.coordinate2adjacent(points)
        c = cu.get_from_qasm(qasm)
        # c.draw('mpl').show()
        simulator = AerSimulator()
        avr = 0
        for i in range(repeat):
            try:
                ct = transpile(circuits=c, coupling_map=adj_list, optimization_level=1,backend=simulator)
                d = ct.depth()
                avr += d
                result.append(d)
            except Exception as e:
                traceback.print_exc()
        result.append(avr/repeat)
        return  result

    @staticmethod
    def get_qiskit_fidelity(qasm:str):
        adj_list = pu.coordinate2adjacent(points)
        circuit = cu.get_from_qasm(qasm)
        fidelity = 0
        for i in range(10):
            fidelity += circuit_fidelity_benchmark(circuit=circuit,coupling_map=adj_list,type='qiskit')
        return fidelity/10

    @staticmethod
    def get_fidelity(qasm:str,matrix):
        circuit = CircutUtil.get_from_qasm(qasm)
        G = nx.DiGraph()
        # 添加节点
        num_nodes = len(matrix)
        G.add_nodes_from(range(num_nodes))

        # 添加边
        for i in range(num_nodes):
            for j in range(num_nodes):
                if matrix[i][j] == 1:
                    G.add_edge(i, j)

        adj_list = GraphUtil.get_adj_list(G)
        init_layout = list(range(len(circuit.qubits)))
        fidelity = 0
        for i in range(10):
            fidelity += circuit_fidelity_benchmark(circuit,coupling_map=adj_list,type='rl',initial_layout = init_layout)
        return fidelity/10
    #用于测试模型运行后的结果
    @staticmethod
    def test_result(qasm,matrix = None, array = None):
        #test rl and mix
        if matrix is None:
            matrix =gu.restore_from_1d_array(array)
        res = gu.test_adj_matrix(matrix,qasm)
        print(res)
        mean = np.mean(res, axis=0)
        rl = mean[0]
        mix = mean[1]
        qiskit = np.mean(Benchmark.get_qiskit_depth(qasm))
        print('rl = %r,qiskit = %r , mix = %r:'%(rl,qiskit,mix))

    #比较保真度
    @staticmethod
    def test_fidelity(qasm,matrix = None, array = None):
        if matrix is None:
            matrix =gu.restore_from_1d_array(array)
        #qiskit
        f1= Benchmark.get_qiskit_fidelity(qasm)
        #rl
        f2= Benchmark.get_fidelity(qasm,matrix)
        print(f"q_fidelity={f1}\nr_fidelity={f2}")


    @staticmethod
    def compare_gates(qasm,matrix = None, array = None,bits=0):
        data = [qasm]
        circuit = cu.get_from_qasm(qasm)
        simulator = AerSimulator()
        qt = transpile(circuits=circuit, coupling_map=pu.coordinate2adjacent(points), optimization_level=3, backend=simulator)
        qt = remove_idle_qwires(qt)
        #qt.decompose().draw('mpl').show()
        qiskit_opts = qt.decompose().count_ops()
        depth = qt.decompose().depth()
        #print(sorted(qiskit_opts.items()))

        qgates_cnt= sum(qiskit_opts.values())
        q_idle_rate= (1-qgates_cnt/(bits*depth)).__round__(4)
        data.append(qgates_cnt)
        data.append(depth)
        data.append(q_idle_rate)
        print(f'qiskit gates = {qgates_cnt},qiskit_depth={depth}, q_idle_rate = {q_idle_rate}')
        ###################

        if matrix is None:
            matrix =gu.restore_from_1d_array(array)
        circuit = cu.get_from_qasm(qasm)
        G = nx.DiGraph()
        # 添加节点
        num_nodes = len(matrix)
        G.add_nodes_from(range(num_nodes))

        # 添加边
        for i in range(num_nodes):
            for j in range(num_nodes):
                if matrix[i][j] == 1:
                    G.add_edge(i, j)
        layout = list(range(len(circuit.qubits)))
        rt = transpile(circuits=circuit, coupling_map=GraphUtil.get_adj_list(G), initial_layout=layout, optimization_level=3, backend=simulator)
        #rt.decompose().draw('mpl').show()
        rl_opts = rt.decompose().count_ops()
        depth = rt.decompose().depth()
        gates_cnt = sum(rl_opts.values())
        r_idle_rate = (1 - (gates_cnt / (bits * depth))).__round__(4)

        data.append(gates_cnt)
        data.append(depth)
        data.append(r_idle_rate)
        print(f'rl gates = {gates_cnt}rl_depth={depth},r_idle_rate = {r_idle_rate}')

        #write data
        #CSVUtil.append_data(r'D:\workspace\data\benchmark\idle_reate.csv',[data])
        improvement = ((q_idle_rate-r_idle_rate)/q_idle_rate).__round__(4)*100
        improvement = improvement.__round__(2)
        print(f'& {data[1]} & {data[2]} & {data[3]}  & {data[4]}  & {data[5]}  & {data[6]}  & {improvement} $\downarrow$ ')
def remove_idle_qwires(circ):
    dag = circuit_to_dag(circ)

    idle_wires = list(dag.idle_wires())
    for w in idle_wires:
        dag._remove_idle_wire(w)
        dag.qubits.remove(w)

    dag.qregs = OrderedDict()
    return dag_to_circuit(dag)

if __name__ == '__main__':
    array  =[1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1]


    #    [0, 1, 0, 1, 1, 0, 1, 0],
    #    [0, 0, 1, 0, 1, 1, 0, 1],
    #    [1, 0, 1, 1, 0, 1, 0, 0],
    #    [0, 1, 0, 1, 1, 0, 1, 0],
    #    [1, 0, 1, 0, 0, 1, 0, 1],
    #    [1, 1, 0, 1, 0, 0, 1, 0]]
    qasm = 'real_amp/realamprandom_indep_qiskit_9qasm'
    # start_time = time.time()
    # for i in range(3):
    #     Benchmark.test_result(matrix = None,array=array,qasm=qasm)
    # print(time.time() - start_time)
    #Benchmark.test_fidelity(qasm,array=array,matrix=None)
    #print(Benchmark.get_qiskit_depth('random/random_indep_qiskit_14.qasm'))
    Benchmark.compare_gates(qasm=qasm,array=array,bits = 10)