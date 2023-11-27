import os

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import traceback

from utils.file_util import FileUtil

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

    #10W 次compiled_circuit.decompose().depth() 在 笔记本上大概需要30s
    @staticmethod
    def get_circuit_score(circuit:QuantumCircuit, adj:list,initial_layout: list) -> int:
        try:
            compiled_circuit = transpile(circuits=circuit, seed_transpiler=1234,coupling_map=adj, initial_layout=initial_layout, backend=simulator)
            #return compiled_circuit.depth()
            d_circuit = compiled_circuit.decompose()
            return d_circuit.depth()
        except:
            return None


    @staticmethod
    def get_circuit_score1(circuit:QuantumCircuit, adj:list) -> int:
        #[0,1,2,....,qubits的个数-1]
        layout = list(range(len(circuit.qubits)))
        try:
            compiled_circuit = transpile(circuits=circuit, seed_transpiler=1234,coupling_map=adj,initial_layout=layout, optimization_level=1,backend=simulator)
            #return compiled_circuit.depth()
            d_circuit = compiled_circuit.decompose()
            return d_circuit.depth()
        except Exception as e:
            #print(adj)
            #traceback.print_exc()
            return None

    @staticmethod
    #获取操作数量之和（包括所有类型） #1W 次compiled_circuit.decompose().depth() 在 笔记本上大概需要 50s
    def ops_cnt(circuit:QuantumCircuit,adj:list,initial_layout: list):
        try:
            compiled_circuit = transpile(circuits=circuit, coupling_map=adj, initial_layout=initial_layout,
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


def main():
    # Add a H gate on qubit 0
    # circuit.h(0)
    circuit = QuantumCircuit(5)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    circuit.cx(0, 4)

    # Map the quantum measurement to the classical bits
    # circuit.measure([0, 1], [0, 1])

    # adj = [[0, 1],[1, 2],[2, 3],[3, 4],]

    # 全连接
    # adj = [[0, 1], [0, 3], [1, 0], [1, 2], [1, 4], [2, 1],
    #        [2, 5], [3, 0], [3, 4], [3, 6], [4, 1], [4, 3], [4, 5],
    #        [4, 7], [5, 2], [5, 4], [5, 8], [6, 3], [6, 7], [7, 4],
    #        [7, 6], [7, 8], [8, 5], [8, 7]]
    cu = CircutUtil('')
    adj = cu.coordinate2adjacent([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)])
    qr = circuit.qubits
    initial_layout = [None,qr[1],  None,qr[2], qr[0], qr[3],None, qr[4],   None]
    #initial_layout = [None, qr[1], None, qr[2], qr[0], qr[3], None, qr[4], None]
    compiled_circuit = transpile(circuits=circuit,
                                # initial_layout=initial_layout,
                                 coupling_map=adj,
                                 backend=simulator)

    #compiled_circuit.draw('mpl').show()
    #print(compiled_circuit.decompose().depth())

    print(adj)
    cu.get_circuit_score1(circuit,adj =adj )

    compiled_circuit = transpile(circuits=circuit, coupling_map=adj, initial_layout=initial_layout,
                                 backend=simulator, seed_transpiler=1234)
    d_circuit = compiled_circuit.decompose()
    d_circuit.draw('mpl').show()
    print(d_circuit.depth())


if __name__ == '__main__':
    #main()
    pass