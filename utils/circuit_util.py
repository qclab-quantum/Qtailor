import os
import traceback

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from utils.file.file_util import FileUtil

simulator = AerSimulator()

class CircutUtil:
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
        initial_layout = list(range(len(circuit.qubits)))
        try:
            avr = 0
            for i in range(3):
                cc = transpile(circuits=circuit, coupling_map=adj,initial_layout=initial_layout,layout_method='sabre',routing_method='sabre', optimization_level=1,backend=simulator)
                #avr += cc.size() * 0.5 + cc.depth()*0.5
                avr += cc.decompose().depth()

            #取平均值
            return avr/3
        except Exception as e:
            #traceback.print_exc()
            return None

    @staticmethod
    def count_gates(type,circuit:QuantumCircuit, adj, gates=[],) -> int:
        try:
            if type == 'rl':
                compiled_circuit = transpile(circuits=circuit,
                                             coupling_map=adj,
                                             initial_layout=list(range(len(circuit.qubits))),
                                             backend=simulator)
            elif type == 'qiskit':
                compiled_circuit = transpile(circuits=circuit,
                                             coupling_map=adj,
                                             backend=simulator)

            ops = compiled_circuit.count_ops()
            if len(gates) == 0:
                return  sum(ops.values())
            else:
                return  sum(ops[g] for g in gates if g in ops)
        except Exception as e:
            traceback.print_exc()
            return 999999

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
    # qasm to qiskit circuit object
    def get_from_qasm(name:str):
        qasm_str = FileUtil.read_all('benchmark'+ os.path.sep+name)
        circuit = QuantumCircuit.from_qasm_str(qasm_str=qasm_str)
        return circuit

    @staticmethod
    def draw_circult(qasm):
        circuit = CircutUtil.get_from_qasm(qasm)
        circuit.draw('mpl').show()

#获取正态分布数组
def get_random_std_arr(n, mean, std=1):
    random_array = np.random.normal(mean, std, n)
    rounded_array = np.rint(random_array).astype(int)
    final_array = np.clip(rounded_array, 0, n)
    return final_array

#生成随机线路
def generate_circuit(n,p):
    circuit = QuantumCircuit(n)
    random_array = get_random_std_arr(n=(p*n),mean=(n/2))
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
    for i in [80]:
        for j in [3,4]:
            generate_circuit(i,j)

