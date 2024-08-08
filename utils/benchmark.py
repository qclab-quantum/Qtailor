import time
import traceback
from collections import OrderedDict
import matplotlib
matplotlib.use('TkAgg')
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
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

t1 = 50000
t2 = 70000
class Benchmark():

    def __init__(self,qasm ):
        self.qasm = qasm

    '''
    file_path: 'the path of csv file ', is file_path is not None, the result will be saved in csv file
    matrix:  the adjacency matrix representing the graph 
    qasm:  the qasm file path of circuits, qasm file is in benchmark folder
    draw:  true =  draw the topology(graph)
    show_in_html: show topology in html  
    '''
    @staticmethod
    def depth_benchmark(file_path,matrix:np.ndarray,qasm:str,draw=False,show_in_html=False):
        b_1 = Benchmark.get_rl_depth(matrix, qasm)
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
        if file_path:
            CSVUtil.append_data(file_path, data)
        if draw:
            gu.draw_adj_matrix(matrix,is_draw_nt=True)
            #pu.plot_points(points)
        return rl,qiskit,mix

    @staticmethod
    def get_rl_depth(adj_matrix, qasm):
        simulator = AerSimulator()
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
                ct1 = transpile(circuits=circuit, coupling_map=adj_list, initial_layout=layout,  optimization_level=1, backend=simulator)
                ct2 = transpile(circuits=circuit, coupling_map=adj_list, optimization_level=3, backend=simulator)
                d1 = ct1.depth()
                d2 = ct2.depth()
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
    def gates_benchmark(file_path,matrix:np.ndarray,qasm:str,draw=False,show_in_html=False):
        rl =  Benchmark.get_rl_gates(matrix, qasm)
        #get Qiskit result
        qiskit = Benchmark.get_qiskit_gates(qasm)
        print('rl = %r,qiskit= %r,'%(rl,qiskit,))

        #write to csv file
        data = []
        for i in range(len(rl)):
            data.append(['','',rl[i],qiskit[i]])
        if file_path:
            CSVUtil.append_data(file_path, data)
        if draw:
            gu.draw_adj_matrix(matrix,is_draw_nt=True)
            #pu.plot_points(points)
        return rl[-1],qiskit[-1]

    @staticmethod
    def get_rl_gates(matrix:np.ndarray,qasm:str,draw=False,show_in_html=False):
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

        adj = GraphUtil.get_adj_list(G)
        gates_cnt = 0
        result = []
        repeat = 20
        for i in range(repeat):
            try:
                cnt = CircutUtil.count_gates(type='rl',circuit = circuit,adj=adj,gates=['cp','cx','swap'])
                gates_cnt += cnt
                result.append(cnt)
            except Exception as e:
                print(e)
                result.append(-1)
        result.append(gates_cnt/repeat)
        return result

    @staticmethod
    def get_qiskit_gates(qasm:str):

        adj= pu.coordinate2adjacent(points)
        circuit = cu.get_from_qasm(qasm)

        gates_cnt = 0
        result = []
        repeat = 20
        for i in range(repeat):
            cnt = CircutUtil.count_gates(type='qiskit', circuit = circuit,adj=adj,gates=['cp','cx','swap'])
            gates_cnt += cnt
            result.append(cnt)
        result.append(gates_cnt/repeat)
        return  result

    @staticmethod
    def get_qiskit_fidelity(qasm:str):
        adj_list = pu.coordinate2adjacent(points)
        circuit = cu.get_from_qasm(qasm)
        fidelity = 0
        repeat = 1
        for i in range(repeat):
            fidelity += circuit_fidelity_benchmark(circuit=circuit,coupling_map=adj_list,type='qiskit',t1=t1,t2=t2)
        return fidelity/repeat

    #用于测试模型运行后的结果
    @staticmethod
    def test_result(qasm,matrix = None, array = None):
        #test rl and mix
        if matrix is None:
            matrix =gu.restore_from_1d_array(array)
        res = CircutUtil.get_rl_depth(matrix,qasm)
        print(res)
        mean = np.mean(res, axis=0)
        rl = mean[0]
        mix = mean[1]
        qiskit = np.mean(Benchmark.get_qiskit_depth(qasm))
        print('rl = %r,qiskit = %r , mix = %r:'%(rl,qiskit,mix))

    #比较保真度
    @staticmethod
    def test_fidelity(qasm,matrix = None):
        #qiskit
        f1= Benchmark.get_qiskit_fidelity(qasm)
        #rl
        f2= Benchmark.get_fidelity(qasm,matrix)
        #print(f"q_fidelity={f1}\n r_fidelity={f2}")
        return f1,f2

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
        fidelity = 0
        repeat = 1
        for i in range(repeat):
            fidelity += circuit_fidelity_benchmark(circuit,coupling_map=adj_list,type='rl',t1=t1,t2=t2)
        return fidelity/repeat

    @staticmethod
    def compare_gates(qasm,matrix = None, array = None,bits=0):
        data = [qasm]
        circuit = cu.get_from_qasm(qasm)
        simulator = AerSimulator()
        qt = transpile(circuits=circuit, coupling_map=pu.coordinate2adjacent(points), optimization_level=3, backend=simulator)
        qt = remove_idle_qwires(qt)
        #qt.decompose().draw('mpl').show()
        qiskit_opts = qt.decompose().count_ops()
        depth = qt.depth()
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
        depth = rt.depth()
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

def benchmark0808():
    array = [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    reshape_obs = GraphUtil.restore_from_1d_array(array)
    qasm = 'qft/qft_indep_qiskit_20.qasm'
    csv_path = 'd:/temp/qft.csv'
    rl, qiskit = Benchmark.gates_benchmark(csv_path, reshape_obs, qasm, False)

def rount_arr(arr):
    return [[round(element, 2) for element in row] for row in arr]
if __name__ == '__main__':
    topology = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    matrix = gu.restore_from_1d_array(topology)
    qasm = 'real_amp/realamprandom_indep_qiskit_6.qasm'

    #Benchmark.depth_benchmark(file_path=None,matrix=matrix,qasm=qasm,draw=True,show_in_html=True)
    result = np.full((20, 19), 1.0)
    f1_all=np.full((20, 19), 1.0)
    f2_all=np.full((20, 19), 1.0)
    t2_time=np.full((20, 19), 1.0)

    t1max=50e3
    for i in range(20):
        t1=t1max * (0.2+i*0.04)
        t1_arr = [].append(t1)
        t2max=1.2 * t1

        for j in range(19):
            t2=t2max * ((j+1) * 0.05)
            #print(t1, t2)
            f1,f2=Benchmark.test_fidelity(qasm,matrix=matrix)
            improve = (f2-f1)/f2
            f1_all[i][j]=f1.__round__(2)
            f2_all[i][j]=f2.__round__(2)
            result[i][j] = improve.__round__(2)*100
            t2_time[i][j]=t2.__round__(2)
    print(np.array2string(f1_all, separator=', '))
    print('f1')
    print(np.array2string(f2_all, separator=', '))
    print('f2')
    print(np.array2string(result, separator=', '))
    print('result')
    print(np.array2string(t2_time, separator=', '))
    print('t2')



    #
    # print(Benchmark.get_qiskit_gates(qasm))
    # print(Benchmark.get_rl_gates(matrix,qasm))
