from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

simulator = AerSimulator()

# get gates from compiled circuit
def get_compiled_gate(circuit:QuantumCircuit, adj:list,initial_layout: list) -> int:

    try:
        compiled_circuit = transpile(circuits=circuit, coupling_map=adj, initial_layout=initial_layout, backend=simulator)
    except:
        return -1
    return compiled_circuit.num_nonlocal_gates()

if __name__ == '__main__':
    circuit = QuantumCircuit(3, 2)
    # Add a H gate on qubit 0
    #circuit.h(0)
    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
    circuit.swap(1,0)
    # Map the quantum measurement to the classical bits
    #circuit.measure([0, 1], [0, 1])


    adj = [
        [0, 1],
        [1, 0],
        [2, 3]

    ]
    #circuit.decompose('CNOT').draw('mpl').show()
    #circuit.draw('mpl').show()
    compiled_circuit = transpile(circuits=circuit, coupling_map=adj, backend=simulator)
    compiled_circuit.decompose().draw('mpl').show()
