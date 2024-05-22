import os

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from utils.file.file_util import FileUtil

simulator = AerSimulator()

class CircutUtil:
    def __init__(self, name):
        self.name = name
    # 获取线路深度（门分解后）
    @staticmethod
    def get_circuit_depth(circuit:QuantumCircuit, adj:list,initial_layout: list) -> int:
        try:
            compiled_circuit = transpile(circuits=circuit, coupling_map=adj, initial_layout=initial_layout, backend=simulator)
        except:
            return -1
        return compiled_circuit.decompose().depth()


    @staticmethod
    def get_circuit_score(circuit:QuantumCircuit, adj:list) -> int:
        #[0,1,2,....,qubits的个数-1]
        layout = list(range(len(circuit.qubits)))
        try:
            avr = 0
            for i in range(3):
                cc = transpile(circuits=circuit, coupling_map=adj,initial_layout=layout,layout_method='sabre',routing_method='sabre', optimization_level=1,backend=simulator)
                #avr += cc.size() * 0.5 + cc.depth()*0.5
                avr += cc.decompose().depth()

            #取平均值
            return avr/3
        except Exception as e:
            #print(adj)
            #traceback.print_exc()
            return None

    @staticmethod

    def get_gates(circuit:QuantumCircuit,adj:list,initial_layout: list):
        try:
            compiled_circuit = transpile(circuits=circuit,
                                         coupling_map=adj,
                                         initial_layout=initial_layout,
                                         backend=simulator)
            ops = compiled_circuit.count_ops()
            return  sum(ops.values())
        except:
            return None
        return compiled_circuit.decompose().depth()



    # get adj  from coordinate
    @staticmethod
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

    @staticmethod
    def adjacency2matrix(adj_list):
        max_index = max(max(pair) for pair in adj_list)
        matrix = [[0] * (max_index + 1) for _ in range(max_index + 1)]

        for pair in adj_list:
            matrix[pair[0]][pair[1]] = 1
            matrix[pair[1]][pair[0]] = 1

        return matrix

    @staticmethod
    #从 qasm 文件中读取 代码并转为 qiskit circuit 对象
    def get_from_qasm(name:str):
        qasm_str = FileUtil.read_all('benchmark'+ os.path.sep+name)
        circuit = QuantumCircuit.from_qasm_str(qasm_str=qasm_str)
        return circuit

    @staticmethod
    def draw_circult(qasm):
        circuit = CircutUtil.get_from_qasm(qasm)
        circuit.draw('mpl').show()

def generate_random_integer(n, mean, std=1):
    random_array = np.random.normal(mean, std, n)
    rounded_array = np.rint(random_array).astype(int)
    final_array = np.clip(rounded_array, 0, n)
    #print(final_array)
    return final_array

#生成随机线路
def generate_circuit(n,p):
    circuit = QuantumCircuit(n)
    random_array = generate_random_integer(n=(p*n),mean=(n/2))
    i = 0
    while (i+1) < len(random_array):
        ctrl = random_array[i]
        target = random_array[i+1]
        i = i + 1
        if ctrl == target:
            continue
        circuit.cx(ctrl,target)

    file_path = os.path.join(FileUtil.get_root_dir(),'benchmark','custom',str(n)+'_'+str(p * n)+'.qasm')
    print('\''+'custom/',str(n)+'_'+str(p * n)+'.qasm'+'\',')
    FileUtil.write(file_path, circuit.qasm())


if __name__ == '__main__':
    #CircutUtil.draw_circult('qnn/qnn_indep_qiskit_3.qasm')
    for i in [50,60,70,80,90,100]:
        for j in [10]:
            generate_circuit(i,j)