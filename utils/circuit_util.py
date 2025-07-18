import os
import traceback
from typing import Optional, List, Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from utils.file.file_util import FileUtil

# 使用单例模式避免重复创建simulator
_simulator = None

def get_simulator():
    global _simulator
    if _simulator is None:
        _simulator = AerSimulator()
    return _simulator

class CircuitUtil:  # 修正类名拼写错误
    """量子电路工具类"""
    
    @staticmethod
    def get_circuit_depth(circuit: QuantumCircuit, adj: List[List[int]], initial_layout: List[int]) -> int:
        """
        获取线路深度（门分解后）
        
        Args:
            circuit: 量子电路
            adj: 邻接列表
            initial_layout: 初始布局
            
        Returns:
            电路深度，失败时返回-1
        """
        try:
            compiled_circuit = transpile(
                circuits=circuit, 
                coupling_map=adj, 
                initial_layout=initial_layout, 
                backend=get_simulator()
            )
            return compiled_circuit.decompose().depth()
        except Exception as e:
            print(f"获取电路深度失败: {e}")
            return -1

    @staticmethod
    def get_circuit_score(circuit: QuantumCircuit, adj: List[List[int]], iterations: int = 3) -> Optional[float]:
        """
        获取电路评分
        
        Args:
            circuit: 量子电路
            adj: 邻接列表
            iterations: 迭代次数，默认3次
            
        Returns:
            平均深度评分，失败时返回None
        """
        initial_layout = list(range(len(circuit.qubits)))
        try:
            total_depth = 0
            for _ in range(iterations):
                compiled_circuit = transpile(
                    circuits=circuit, 
                    coupling_map=adj,
                    initial_layout=initial_layout,
                    layout_method='sabre',
                    routing_method='sabre', 
                    optimization_level=1,
                    backend=get_simulator()
                )
                total_depth += compiled_circuit.decompose().depth()
            
            return total_depth / iterations
        except Exception as e:
            print(f"获取电路评分失败: {e}")
            return None

    @staticmethod
    def count_gates(transpile_type: str, circuit: QuantumCircuit, adj: List[List[int]], 
                   gates: Optional[List[str]] = None) -> int:
        """
        统计门的数量
        
        Args:
            transpile_type: 编译类型 ('rl' 或 'qiskit')
            circuit: 量子电路
            adj: 邻接列表
            gates: 要统计的门类型列表，为空时统计所有门
            
        Returns:
            门的总数，失败时返回999999
        """
        if gates is None:
            gates = []
            
        try:
            if transpile_type == 'rl':
                compiled_circuit = transpile(
                    circuits=circuit,
                    coupling_map=adj,
                    initial_layout=list(range(len(circuit.qubits))),
                    backend=get_simulator()
                )
            elif transpile_type == 'qiskit':
                compiled_circuit = transpile(
                    circuits=circuit,
                    coupling_map=adj,
                    backend=get_simulator()
                )
            else:
                raise ValueError(f"不支持的编译类型: {transpile_type}")

            ops = compiled_circuit.count_ops()
            
            if not gates:
                return sum(ops.values())
            else:
                return sum(ops.get(gate, 0) for gate in gates)
                
        except Exception as e:
            print(f"统计门数量失败: {e}")
            traceback.print_exc()
            return 999999

    @staticmethod
    def coordinate2adjacent(points: List[List[Union[int, float]]]) -> List[List[int]]:
        """
        根据坐标点生成邻接列表
        
        Args:
            points: 坐标点列表
            
        Returns:
            邻接列表
        """
        import math
        
        adjacency_dict = {}
        num_points = len(points)
        
        for i in range(num_points):
            adjacent_points = []
            for j in range(num_points):
                if i != j:
                    distance = math.sqrt(
                        (points[i][0] - points[j][0])**2 + 
                        (points[i][1] - points[j][1])**2
                    )
                    if abs(distance - 1.0) < 1e-10:  # 使用浮点数比较
                        adjacent_points.append(j)
            adjacency_dict[i] = adjacent_points

        # 转换为qiskit格式
        result = []
        for node, neighbors in adjacency_dict.items():
            for neighbor in neighbors:
                result.append([node, neighbor])
        
        return result

    @staticmethod
    def adjacency2matrix(adj_list: List[List[int]]) -> List[List[int]]:
        """
        将邻接列表转换为邻接矩阵
        
        Args:
            adj_list: 邻接列表
            
        Returns:
            邻接矩阵
        """
        if not adj_list:
            return []
            
        max_index = max(max(pair) for pair in adj_list)
        matrix = [[0] * (max_index + 1) for _ in range(max_index + 1)]

        for pair in adj_list:
            matrix[pair[0]][pair[1]] = 1
            matrix[pair[1]][pair[0]] = 1

        return matrix

    @staticmethod
    def get_from_qasm(name: str) -> QuantumCircuit:
        """
        从QASM文件创建量子电路对象
        
        Args:
            name: QASM文件名
            
        Returns:
            量子电路对象
        """
        qasm_path = os.path.join('benchmark', name)
        qasm_str = FileUtil.read_all(qasm_path)
        return QuantumCircuit.from_qasm_str(qasm_str=qasm_str)

    @staticmethod
    def draw_circuit(qasm_file: str) -> None:
        """
        绘制电路图
        
        Args:
            qasm_file: QASM文件名
        """
        circuit = CircuitUtil.get_from_qasm(qasm_file)
        circuit.draw('mpl').show()


def get_random_std_arr(n: int, mean: float, std: float = 1.0) -> np.ndarray:
    """
    获取正态分布数组
    
    Args:
        n: 数组长度
        mean: 均值
        std: 标准差
        
    Returns:
        正态分布的整数数组
    """
    random_array = np.random.normal(mean, std, n)
    rounded_array = np.rint(random_array).astype(int)
    final_array = np.clip(rounded_array, 0, n - 1)  # 修正边界条件
    return final_array


def generate_circuit(n: int, p: float) -> None:
    """
    生成随机量子电路
    
    Args:
        n: 量子比特数
        p: 参数p
    """
    circuit = QuantumCircuit(n)
    num_gates = int(p * n)
    random_array = get_random_std_arr(n=num_gates * 2, mean=(n / 2))
    
    # 成对处理随机数组
    for i in range(0, len(random_array) - 1, 2):
        ctrl = random_array[i]
        target = random_array[i + 1]
        
        if ctrl != target:
            circuit.cx(ctrl, target)

    # 使用更清晰的文件路径构建
    filename = f"{n}_{num_gates}.qasm"
    file_path = os.path.join(
        FileUtil.get_root_dir(), 
        'benchmark', 
        'custom', 
        filename
    )
    
    print(f"'custom/{filename}',")
    FileUtil.write(file_path, circuit.qasm())


if __name__ == '__main__':
    # 使用更清晰的参数名
    qubit_counts = [80]
    p_values = [3, 4]
    
    for qubits in qubit_counts:
        for p in p_values:
            generate_circuit(qubits, p)

