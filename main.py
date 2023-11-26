from qiskit import QuantumCircuit, transpile

from utils.circuit_util import CircutUtil

circuit = QuantumCircuit.from_qasm_str("""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg meas[3];
ry(0.416499152901678) q[0];
ry(1.6289300606876) q[1];
ry(-0.301497213523459) q[2];
cx q[1],q[2];
cx q[0],q[1];
ry(-2.66473611257652) q[0];
ry(-1.88697596497767) q[1];
ry(1.42601786989662) q[2];
cx q[1],q[2];
cx q[0],q[1];
ry(-0.770603706024532) q[0];
ry(-2.33750437070044) q[1];
ry(3.04370801572893) q[2];
barrier q[0],q[1],q[2];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
""")
circuit.draw('mpl').show()

from qiskit_aer import AerSimulator
import traceback
simulator = AerSimulator()
cu = CircutUtil('')
#adj = [ [0, 1], [0, 2], [1, 0],  [1, 2], [2, 0], [2, 1], [2, 3],[2, 4]]
adj = [ [0, 1], [0, 2]]
initial_layout =[0,1,2,3,4,5]
compiled_circuit = transpile(circuits=circuit,
                             # initial_layout=initial_layout,
                             coupling_map=adj,
                             backend=simulator,
                             optimization_level=0)

compiled_circuit.draw('mpl').show()
print(compiled_circuit.decompose().depth())

#print(cu.get_circuit_score1(circuit, adj=adj))