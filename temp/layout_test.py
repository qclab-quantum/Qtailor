from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Use Aer's AerSimulator
simulator = AerSimulator()

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(3, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)
circuit.cx(0, 2)
# Map the quantum measurement to the classical bits
circuit.measure([0, 1], [0, 1])

# Compile the circuit for the support instruction set (basis_gates)
# and topology (coupling_map) of the backend
adj = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4]
]
compiled_circuit = transpile(circuits=circuit, coupling_map=adj,initial_layout=[0, 1,2],backend =simulator)


# # Execute the circuit on the aer simulator
# job = simulator.run(compiled_circuit, shots=1000)
#
# # Grab results from the job
# result = job.result()
#
# # Returns counts
# counts = result.get_counts(compiled_circuit)
# print("\nTotal count for 00 and 11 are:", counts)

# Draw the circuit
circuit.draw("mpl").show()
compiled_circuit.draw("mpl").show()
print(circuit.num_nonlocal_gates())
print(compiled_circuit.num_nonlocal_gates())