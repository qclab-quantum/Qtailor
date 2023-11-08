from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

simulator = AerSimulator()

# 获取线路深度（门分解后）
def get_circuit_depth(circuit:QuantumCircuit, adj:list,initial_layout: list) -> int:
    try:
        compiled_circuit = transpile(circuits=circuit, coupling_map=adj, initial_layout=initial_layout, backend=simulator)
    except:
        return -1
    return compiled_circuit.decompose().depth()

if __name__ == '__main__':

    # Add a H gate on qubit 0
    #circuit.h(0)
    circuit = QuantumCircuit(5)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    circuit.cx(0, 4)

    # Map the quantum measurement to the classical bits
    #circuit.measure([0, 1], [0, 1])

    adj = [[0, 1],[1, 2],[2, 3],[3, 4],]

    #全连接
    # adj = [[0, 1], [0, 3], [1, 0], [1, 2], [1, 4], [2, 1],
    # [2, 5], [3, 0], [3, 4], [3, 6], [4, 1], [4, 3], [4, 5],
    # [4, 7], [5, 2], [5, 4], [5, 8], [6, 3], [6, 7], [7, 4],
    # [7, 6], [7, 8], [8, 5], [8, 7]]

    compiled_circuit = transpile(circuits=circuit,
                                coupling_map=adj,
                                 backend=simulator)

    #compiled_circuit.decompose().draw('mpl').show()
    compiled_circuit.draw('mpl').show()
    print(compiled_circuit.depth())
