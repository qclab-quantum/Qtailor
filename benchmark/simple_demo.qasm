//用于测试模型性能和收敛性
// 线路 qiskit 代码如下:
//    circuit = QuantumCircuit(5)
//    circuit.cx(0, 1)
//    circuit.cx(0, 2)
//    circuit.cx(0, 3)
//    circuit.cx(0, 4)

OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
cx q[0],q[1];
cx q[0],q[2];
cx q[0],q[3];
cx q[0],q[4];

