from qiskit import QuantumCircuit

circuit = QuantumCircuit(3, 2)
circuit.cx(0, 1)
circuit.cx(0, 2)
circuit.draw(output='mpl').show()