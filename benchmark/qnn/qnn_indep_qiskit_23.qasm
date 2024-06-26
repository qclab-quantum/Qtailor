// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg meas[23];
u2(2.0,-pi) q[0];
u2(2.0,-pi) q[1];
cx q[0],q[1];
p(9.17283818781952) q[1];
cx q[0],q[1];
u2(2.0,-pi) q[2];
cx q[0],q[2];
p(9.17283818781952) q[2];
cx q[0],q[2];
cx q[1],q[2];
p(9.17283818781952) q[2];
cx q[1],q[2];
u2(2.0,-pi) q[3];
cx q[0],q[3];
p(9.17283818781952) q[3];
cx q[0],q[3];
cx q[1],q[3];
p(9.17283818781952) q[3];
cx q[1],q[3];
cx q[2],q[3];
p(9.17283818781952) q[3];
cx q[2],q[3];
u2(2.0,-pi) q[4];
cx q[0],q[4];
p(9.17283818781952) q[4];
cx q[0],q[4];
cx q[1],q[4];
p(9.17283818781952) q[4];
cx q[1],q[4];
cx q[2],q[4];
p(9.17283818781952) q[4];
cx q[2],q[4];
cx q[3],q[4];
p(9.17283818781952) q[4];
cx q[3],q[4];
u2(2.0,-pi) q[5];
cx q[0],q[5];
p(9.17283818781952) q[5];
cx q[0],q[5];
cx q[1],q[5];
p(9.17283818781952) q[5];
cx q[1],q[5];
cx q[2],q[5];
p(9.17283818781952) q[5];
cx q[2],q[5];
cx q[3],q[5];
p(9.17283818781952) q[5];
cx q[3],q[5];
cx q[4],q[5];
p(9.17283818781952) q[5];
cx q[4],q[5];
u2(2.0,-pi) q[6];
cx q[0],q[6];
p(9.17283818781952) q[6];
cx q[0],q[6];
cx q[1],q[6];
p(9.17283818781952) q[6];
cx q[1],q[6];
cx q[2],q[6];
p(9.17283818781952) q[6];
cx q[2],q[6];
cx q[3],q[6];
p(9.17283818781952) q[6];
cx q[3],q[6];
cx q[4],q[6];
p(9.17283818781952) q[6];
cx q[4],q[6];
cx q[5],q[6];
p(9.17283818781952) q[6];
cx q[5],q[6];
u2(2.0,-pi) q[7];
cx q[0],q[7];
p(9.17283818781952) q[7];
cx q[0],q[7];
cx q[1],q[7];
p(9.17283818781952) q[7];
cx q[1],q[7];
cx q[2],q[7];
p(9.17283818781952) q[7];
cx q[2],q[7];
cx q[3],q[7];
p(9.17283818781952) q[7];
cx q[3],q[7];
cx q[4],q[7];
p(9.17283818781952) q[7];
cx q[4],q[7];
cx q[5],q[7];
p(9.17283818781952) q[7];
cx q[5],q[7];
cx q[6],q[7];
p(9.17283818781952) q[7];
cx q[6],q[7];
u2(2.0,-pi) q[8];
cx q[0],q[8];
p(9.17283818781952) q[8];
cx q[0],q[8];
cx q[1],q[8];
p(9.17283818781952) q[8];
cx q[1],q[8];
cx q[2],q[8];
p(9.17283818781952) q[8];
cx q[2],q[8];
cx q[3],q[8];
p(9.17283818781952) q[8];
cx q[3],q[8];
cx q[4],q[8];
p(9.17283818781952) q[8];
cx q[4],q[8];
cx q[5],q[8];
p(9.17283818781952) q[8];
cx q[5],q[8];
cx q[6],q[8];
p(9.17283818781952) q[8];
cx q[6],q[8];
cx q[7],q[8];
p(9.17283818781952) q[8];
cx q[7],q[8];
u2(2.0,-pi) q[9];
cx q[0],q[9];
p(9.17283818781952) q[9];
cx q[0],q[9];
cx q[1],q[9];
p(9.17283818781952) q[9];
cx q[1],q[9];
cx q[2],q[9];
p(9.17283818781952) q[9];
cx q[2],q[9];
cx q[3],q[9];
p(9.17283818781952) q[9];
cx q[3],q[9];
cx q[4],q[9];
p(9.17283818781952) q[9];
cx q[4],q[9];
cx q[5],q[9];
p(9.17283818781952) q[9];
cx q[5],q[9];
cx q[6],q[9];
p(9.17283818781952) q[9];
cx q[6],q[9];
cx q[7],q[9];
p(9.17283818781952) q[9];
cx q[7],q[9];
cx q[8],q[9];
p(9.17283818781952) q[9];
cx q[8],q[9];
u2(2.0,-pi) q[10];
cx q[0],q[10];
p(9.17283818781952) q[10];
cx q[0],q[10];
cx q[1],q[10];
p(9.17283818781952) q[10];
cx q[1],q[10];
cx q[2],q[10];
p(9.17283818781952) q[10];
cx q[2],q[10];
cx q[3],q[10];
p(9.17283818781952) q[10];
cx q[3],q[10];
cx q[4],q[10];
p(9.17283818781952) q[10];
cx q[4],q[10];
cx q[5],q[10];
p(9.17283818781952) q[10];
cx q[5],q[10];
cx q[6],q[10];
p(9.17283818781952) q[10];
cx q[6],q[10];
cx q[7],q[10];
p(9.17283818781952) q[10];
cx q[7],q[10];
cx q[8],q[10];
p(9.17283818781952) q[10];
cx q[8],q[10];
cx q[9],q[10];
p(9.17283818781952) q[10];
cx q[9],q[10];
u2(2.0,-pi) q[11];
cx q[0],q[11];
p(9.17283818781952) q[11];
cx q[0],q[11];
cx q[1],q[11];
p(9.17283818781952) q[11];
cx q[1],q[11];
cx q[2],q[11];
p(9.17283818781952) q[11];
cx q[2],q[11];
cx q[3],q[11];
p(9.17283818781952) q[11];
cx q[3],q[11];
cx q[4],q[11];
p(9.17283818781952) q[11];
cx q[4],q[11];
cx q[5],q[11];
p(9.17283818781952) q[11];
cx q[5],q[11];
cx q[6],q[11];
p(9.17283818781952) q[11];
cx q[6],q[11];
cx q[7],q[11];
p(9.17283818781952) q[11];
cx q[7],q[11];
cx q[8],q[11];
p(9.17283818781952) q[11];
cx q[8],q[11];
cx q[9],q[11];
p(9.17283818781952) q[11];
cx q[9],q[11];
cx q[10],q[11];
p(9.17283818781952) q[11];
cx q[10],q[11];
u2(2.0,-pi) q[12];
cx q[0],q[12];
p(9.17283818781952) q[12];
cx q[0],q[12];
cx q[1],q[12];
p(9.17283818781952) q[12];
cx q[1],q[12];
cx q[2],q[12];
p(9.17283818781952) q[12];
cx q[2],q[12];
cx q[3],q[12];
p(9.17283818781952) q[12];
cx q[3],q[12];
cx q[4],q[12];
p(9.17283818781952) q[12];
cx q[4],q[12];
cx q[5],q[12];
p(9.17283818781952) q[12];
cx q[5],q[12];
cx q[6],q[12];
p(9.17283818781952) q[12];
cx q[6],q[12];
cx q[7],q[12];
p(9.17283818781952) q[12];
cx q[7],q[12];
cx q[8],q[12];
p(9.17283818781952) q[12];
cx q[8],q[12];
cx q[9],q[12];
p(9.17283818781952) q[12];
cx q[9],q[12];
cx q[10],q[12];
p(9.17283818781952) q[12];
cx q[10],q[12];
cx q[11],q[12];
p(9.17283818781952) q[12];
cx q[11],q[12];
u2(2.0,-pi) q[13];
cx q[0],q[13];
p(9.17283818781952) q[13];
cx q[0],q[13];
cx q[1],q[13];
p(9.17283818781952) q[13];
cx q[1],q[13];
cx q[2],q[13];
p(9.17283818781952) q[13];
cx q[2],q[13];
cx q[3],q[13];
p(9.17283818781952) q[13];
cx q[3],q[13];
cx q[4],q[13];
p(9.17283818781952) q[13];
cx q[4],q[13];
cx q[5],q[13];
p(9.17283818781952) q[13];
cx q[5],q[13];
cx q[6],q[13];
p(9.17283818781952) q[13];
cx q[6],q[13];
cx q[7],q[13];
p(9.17283818781952) q[13];
cx q[7],q[13];
cx q[8],q[13];
p(9.17283818781952) q[13];
cx q[8],q[13];
cx q[9],q[13];
p(9.17283818781952) q[13];
cx q[9],q[13];
cx q[10],q[13];
p(9.17283818781952) q[13];
cx q[10],q[13];
cx q[11],q[13];
p(9.17283818781952) q[13];
cx q[11],q[13];
cx q[12],q[13];
p(9.17283818781952) q[13];
cx q[12],q[13];
u2(2.0,-pi) q[14];
cx q[0],q[14];
p(9.17283818781952) q[14];
cx q[0],q[14];
cx q[1],q[14];
p(9.17283818781952) q[14];
cx q[1],q[14];
cx q[2],q[14];
p(9.17283818781952) q[14];
cx q[2],q[14];
cx q[3],q[14];
p(9.17283818781952) q[14];
cx q[3],q[14];
cx q[4],q[14];
p(9.17283818781952) q[14];
cx q[4],q[14];
cx q[5],q[14];
p(9.17283818781952) q[14];
cx q[5],q[14];
cx q[6],q[14];
p(9.17283818781952) q[14];
cx q[6],q[14];
cx q[7],q[14];
p(9.17283818781952) q[14];
cx q[7],q[14];
cx q[8],q[14];
p(9.17283818781952) q[14];
cx q[8],q[14];
cx q[9],q[14];
p(9.17283818781952) q[14];
cx q[9],q[14];
cx q[10],q[14];
p(9.17283818781952) q[14];
cx q[10],q[14];
cx q[11],q[14];
p(9.17283818781952) q[14];
cx q[11],q[14];
cx q[12],q[14];
p(9.17283818781952) q[14];
cx q[12],q[14];
cx q[13],q[14];
p(9.17283818781952) q[14];
cx q[13],q[14];
u2(2.0,-pi) q[15];
cx q[0],q[15];
p(9.17283818781952) q[15];
cx q[0],q[15];
cx q[1],q[15];
p(9.17283818781952) q[15];
cx q[1],q[15];
cx q[2],q[15];
p(9.17283818781952) q[15];
cx q[2],q[15];
cx q[3],q[15];
p(9.17283818781952) q[15];
cx q[3],q[15];
cx q[4],q[15];
p(9.17283818781952) q[15];
cx q[4],q[15];
cx q[5],q[15];
p(9.17283818781952) q[15];
cx q[5],q[15];
cx q[6],q[15];
p(9.17283818781952) q[15];
cx q[6],q[15];
cx q[7],q[15];
p(9.17283818781952) q[15];
cx q[7],q[15];
cx q[8],q[15];
p(9.17283818781952) q[15];
cx q[8],q[15];
cx q[9],q[15];
p(9.17283818781952) q[15];
cx q[9],q[15];
cx q[10],q[15];
p(9.17283818781952) q[15];
cx q[10],q[15];
cx q[11],q[15];
p(9.17283818781952) q[15];
cx q[11],q[15];
cx q[12],q[15];
p(9.17283818781952) q[15];
cx q[12],q[15];
cx q[13],q[15];
p(9.17283818781952) q[15];
cx q[13],q[15];
cx q[14],q[15];
p(9.17283818781952) q[15];
cx q[14],q[15];
u2(2.0,-pi) q[16];
cx q[0],q[16];
p(9.17283818781952) q[16];
cx q[0],q[16];
cx q[1],q[16];
p(9.17283818781952) q[16];
cx q[1],q[16];
cx q[2],q[16];
p(9.17283818781952) q[16];
cx q[2],q[16];
cx q[3],q[16];
p(9.17283818781952) q[16];
cx q[3],q[16];
cx q[4],q[16];
p(9.17283818781952) q[16];
cx q[4],q[16];
cx q[5],q[16];
p(9.17283818781952) q[16];
cx q[5],q[16];
cx q[6],q[16];
p(9.17283818781952) q[16];
cx q[6],q[16];
cx q[7],q[16];
p(9.17283818781952) q[16];
cx q[7],q[16];
cx q[8],q[16];
p(9.17283818781952) q[16];
cx q[8],q[16];
cx q[9],q[16];
p(9.17283818781952) q[16];
cx q[9],q[16];
cx q[10],q[16];
p(9.17283818781952) q[16];
cx q[10],q[16];
cx q[11],q[16];
p(9.17283818781952) q[16];
cx q[11],q[16];
cx q[12],q[16];
p(9.17283818781952) q[16];
cx q[12],q[16];
cx q[13],q[16];
p(9.17283818781952) q[16];
cx q[13],q[16];
cx q[14],q[16];
p(9.17283818781952) q[16];
cx q[14],q[16];
cx q[15],q[16];
p(9.17283818781952) q[16];
cx q[15],q[16];
u2(2.0,-pi) q[17];
cx q[0],q[17];
p(9.17283818781952) q[17];
cx q[0],q[17];
cx q[1],q[17];
p(9.17283818781952) q[17];
cx q[1],q[17];
cx q[2],q[17];
p(9.17283818781952) q[17];
cx q[2],q[17];
cx q[3],q[17];
p(9.17283818781952) q[17];
cx q[3],q[17];
cx q[4],q[17];
p(9.17283818781952) q[17];
cx q[4],q[17];
cx q[5],q[17];
p(9.17283818781952) q[17];
cx q[5],q[17];
cx q[6],q[17];
p(9.17283818781952) q[17];
cx q[6],q[17];
cx q[7],q[17];
p(9.17283818781952) q[17];
cx q[7],q[17];
cx q[8],q[17];
p(9.17283818781952) q[17];
cx q[8],q[17];
cx q[9],q[17];
p(9.17283818781952) q[17];
cx q[9],q[17];
cx q[10],q[17];
p(9.17283818781952) q[17];
cx q[10],q[17];
cx q[11],q[17];
p(9.17283818781952) q[17];
cx q[11],q[17];
cx q[12],q[17];
p(9.17283818781952) q[17];
cx q[12],q[17];
cx q[13],q[17];
p(9.17283818781952) q[17];
cx q[13],q[17];
cx q[14],q[17];
p(9.17283818781952) q[17];
cx q[14],q[17];
cx q[15],q[17];
p(9.17283818781952) q[17];
cx q[15],q[17];
cx q[16],q[17];
p(9.17283818781952) q[17];
cx q[16],q[17];
u2(2.0,-pi) q[18];
cx q[0],q[18];
p(9.17283818781952) q[18];
cx q[0],q[18];
cx q[1],q[18];
p(9.17283818781952) q[18];
cx q[1],q[18];
cx q[2],q[18];
p(9.17283818781952) q[18];
cx q[2],q[18];
cx q[3],q[18];
p(9.17283818781952) q[18];
cx q[3],q[18];
cx q[4],q[18];
p(9.17283818781952) q[18];
cx q[4],q[18];
cx q[5],q[18];
p(9.17283818781952) q[18];
cx q[5],q[18];
cx q[6],q[18];
p(9.17283818781952) q[18];
cx q[6],q[18];
cx q[7],q[18];
p(9.17283818781952) q[18];
cx q[7],q[18];
cx q[8],q[18];
p(9.17283818781952) q[18];
cx q[8],q[18];
cx q[9],q[18];
p(9.17283818781952) q[18];
cx q[9],q[18];
cx q[10],q[18];
p(9.17283818781952) q[18];
cx q[10],q[18];
cx q[11],q[18];
p(9.17283818781952) q[18];
cx q[11],q[18];
cx q[12],q[18];
p(9.17283818781952) q[18];
cx q[12],q[18];
cx q[13],q[18];
p(9.17283818781952) q[18];
cx q[13],q[18];
cx q[14],q[18];
p(9.17283818781952) q[18];
cx q[14],q[18];
cx q[15],q[18];
p(9.17283818781952) q[18];
cx q[15],q[18];
cx q[16],q[18];
p(9.17283818781952) q[18];
cx q[16],q[18];
cx q[17],q[18];
p(9.17283818781952) q[18];
cx q[17],q[18];
u2(2.0,-pi) q[19];
cx q[0],q[19];
p(9.17283818781952) q[19];
cx q[0],q[19];
cx q[1],q[19];
p(9.17283818781952) q[19];
cx q[1],q[19];
cx q[2],q[19];
p(9.17283818781952) q[19];
cx q[2],q[19];
cx q[3],q[19];
p(9.17283818781952) q[19];
cx q[3],q[19];
cx q[4],q[19];
p(9.17283818781952) q[19];
cx q[4],q[19];
cx q[5],q[19];
p(9.17283818781952) q[19];
cx q[5],q[19];
cx q[6],q[19];
p(9.17283818781952) q[19];
cx q[6],q[19];
cx q[7],q[19];
p(9.17283818781952) q[19];
cx q[7],q[19];
cx q[8],q[19];
p(9.17283818781952) q[19];
cx q[8],q[19];
cx q[9],q[19];
p(9.17283818781952) q[19];
cx q[9],q[19];
cx q[10],q[19];
p(9.17283818781952) q[19];
cx q[10],q[19];
cx q[11],q[19];
p(9.17283818781952) q[19];
cx q[11],q[19];
cx q[12],q[19];
p(9.17283818781952) q[19];
cx q[12],q[19];
cx q[13],q[19];
p(9.17283818781952) q[19];
cx q[13],q[19];
cx q[14],q[19];
p(9.17283818781952) q[19];
cx q[14],q[19];
cx q[15],q[19];
p(9.17283818781952) q[19];
cx q[15],q[19];
cx q[16],q[19];
p(9.17283818781952) q[19];
cx q[16],q[19];
cx q[17],q[19];
p(9.17283818781952) q[19];
cx q[17],q[19];
cx q[18],q[19];
p(9.17283818781952) q[19];
cx q[18],q[19];
u2(2.0,-pi) q[20];
cx q[0],q[20];
p(9.17283818781952) q[20];
cx q[0],q[20];
cx q[1],q[20];
p(9.17283818781952) q[20];
cx q[1],q[20];
cx q[2],q[20];
p(9.17283818781952) q[20];
cx q[2],q[20];
cx q[3],q[20];
p(9.17283818781952) q[20];
cx q[3],q[20];
cx q[4],q[20];
p(9.17283818781952) q[20];
cx q[4],q[20];
cx q[5],q[20];
p(9.17283818781952) q[20];
cx q[5],q[20];
cx q[6],q[20];
p(9.17283818781952) q[20];
cx q[6],q[20];
cx q[7],q[20];
p(9.17283818781952) q[20];
cx q[7],q[20];
cx q[8],q[20];
p(9.17283818781952) q[20];
cx q[8],q[20];
cx q[9],q[20];
p(9.17283818781952) q[20];
cx q[9],q[20];
cx q[10],q[20];
p(9.17283818781952) q[20];
cx q[10],q[20];
cx q[11],q[20];
p(9.17283818781952) q[20];
cx q[11],q[20];
cx q[12],q[20];
p(9.17283818781952) q[20];
cx q[12],q[20];
cx q[13],q[20];
p(9.17283818781952) q[20];
cx q[13],q[20];
cx q[14],q[20];
p(9.17283818781952) q[20];
cx q[14],q[20];
cx q[15],q[20];
p(9.17283818781952) q[20];
cx q[15],q[20];
cx q[16],q[20];
p(9.17283818781952) q[20];
cx q[16],q[20];
cx q[17],q[20];
p(9.17283818781952) q[20];
cx q[17],q[20];
cx q[18],q[20];
p(9.17283818781952) q[20];
cx q[18],q[20];
cx q[19],q[20];
p(9.17283818781952) q[20];
cx q[19],q[20];
u2(2.0,-pi) q[21];
cx q[0],q[21];
p(9.17283818781952) q[21];
cx q[0],q[21];
cx q[1],q[21];
p(9.17283818781952) q[21];
cx q[1],q[21];
cx q[2],q[21];
p(9.17283818781952) q[21];
cx q[2],q[21];
cx q[3],q[21];
p(9.17283818781952) q[21];
cx q[3],q[21];
cx q[4],q[21];
p(9.17283818781952) q[21];
cx q[4],q[21];
cx q[5],q[21];
p(9.17283818781952) q[21];
cx q[5],q[21];
cx q[6],q[21];
p(9.17283818781952) q[21];
cx q[6],q[21];
cx q[7],q[21];
p(9.17283818781952) q[21];
cx q[7],q[21];
cx q[8],q[21];
p(9.17283818781952) q[21];
cx q[8],q[21];
cx q[9],q[21];
p(9.17283818781952) q[21];
cx q[9],q[21];
cx q[10],q[21];
p(9.17283818781952) q[21];
cx q[10],q[21];
cx q[11],q[21];
p(9.17283818781952) q[21];
cx q[11],q[21];
cx q[12],q[21];
p(9.17283818781952) q[21];
cx q[12],q[21];
cx q[13],q[21];
p(9.17283818781952) q[21];
cx q[13],q[21];
cx q[14],q[21];
p(9.17283818781952) q[21];
cx q[14],q[21];
cx q[15],q[21];
p(9.17283818781952) q[21];
cx q[15],q[21];
cx q[16],q[21];
p(9.17283818781952) q[21];
cx q[16],q[21];
cx q[17],q[21];
p(9.17283818781952) q[21];
cx q[17],q[21];
cx q[18],q[21];
p(9.17283818781952) q[21];
cx q[18],q[21];
cx q[19],q[21];
p(9.17283818781952) q[21];
cx q[19],q[21];
cx q[20],q[21];
p(9.17283818781952) q[21];
cx q[20],q[21];
u2(2.0,-pi) q[22];
cx q[0],q[22];
p(9.17283818781952) q[22];
cx q[0],q[22];
u2(2.0,-pi) q[0];
cx q[1],q[22];
p(9.17283818781952) q[22];
cx q[1],q[22];
u2(2.0,-pi) q[1];
cx q[0],q[1];
p(9.17283818781952) q[1];
cx q[0],q[1];
cx q[2],q[22];
p(9.17283818781952) q[22];
cx q[2],q[22];
u2(2.0,-pi) q[2];
cx q[0],q[2];
p(9.17283818781952) q[2];
cx q[0],q[2];
cx q[1],q[2];
p(9.17283818781952) q[2];
cx q[1],q[2];
cx q[3],q[22];
p(9.17283818781952) q[22];
cx q[3],q[22];
u2(2.0,-pi) q[3];
cx q[0],q[3];
p(9.17283818781952) q[3];
cx q[0],q[3];
cx q[1],q[3];
p(9.17283818781952) q[3];
cx q[1],q[3];
cx q[2],q[3];
p(9.17283818781952) q[3];
cx q[2],q[3];
cx q[4],q[22];
p(9.17283818781952) q[22];
cx q[4],q[22];
u2(2.0,-pi) q[4];
cx q[0],q[4];
p(9.17283818781952) q[4];
cx q[0],q[4];
cx q[1],q[4];
p(9.17283818781952) q[4];
cx q[1],q[4];
cx q[2],q[4];
p(9.17283818781952) q[4];
cx q[2],q[4];
cx q[3],q[4];
p(9.17283818781952) q[4];
cx q[3],q[4];
cx q[5],q[22];
p(9.17283818781952) q[22];
cx q[5],q[22];
u2(2.0,-pi) q[5];
cx q[0],q[5];
p(9.17283818781952) q[5];
cx q[0],q[5];
cx q[1],q[5];
p(9.17283818781952) q[5];
cx q[1],q[5];
cx q[2],q[5];
p(9.17283818781952) q[5];
cx q[2],q[5];
cx q[3],q[5];
p(9.17283818781952) q[5];
cx q[3],q[5];
cx q[4],q[5];
p(9.17283818781952) q[5];
cx q[4],q[5];
cx q[6],q[22];
p(9.17283818781952) q[22];
cx q[6],q[22];
u2(2.0,-pi) q[6];
cx q[0],q[6];
p(9.17283818781952) q[6];
cx q[0],q[6];
cx q[1],q[6];
p(9.17283818781952) q[6];
cx q[1],q[6];
cx q[2],q[6];
p(9.17283818781952) q[6];
cx q[2],q[6];
cx q[3],q[6];
p(9.17283818781952) q[6];
cx q[3],q[6];
cx q[4],q[6];
p(9.17283818781952) q[6];
cx q[4],q[6];
cx q[5],q[6];
p(9.17283818781952) q[6];
cx q[5],q[6];
cx q[7],q[22];
p(9.17283818781952) q[22];
cx q[7],q[22];
u2(2.0,-pi) q[7];
cx q[0],q[7];
p(9.17283818781952) q[7];
cx q[0],q[7];
cx q[1],q[7];
p(9.17283818781952) q[7];
cx q[1],q[7];
cx q[2],q[7];
p(9.17283818781952) q[7];
cx q[2],q[7];
cx q[3],q[7];
p(9.17283818781952) q[7];
cx q[3],q[7];
cx q[4],q[7];
p(9.17283818781952) q[7];
cx q[4],q[7];
cx q[5],q[7];
p(9.17283818781952) q[7];
cx q[5],q[7];
cx q[6],q[7];
p(9.17283818781952) q[7];
cx q[6],q[7];
cx q[8],q[22];
p(9.17283818781952) q[22];
cx q[8],q[22];
u2(2.0,-pi) q[8];
cx q[0],q[8];
p(9.17283818781952) q[8];
cx q[0],q[8];
cx q[1],q[8];
p(9.17283818781952) q[8];
cx q[1],q[8];
cx q[2],q[8];
p(9.17283818781952) q[8];
cx q[2],q[8];
cx q[3],q[8];
p(9.17283818781952) q[8];
cx q[3],q[8];
cx q[4],q[8];
p(9.17283818781952) q[8];
cx q[4],q[8];
cx q[5],q[8];
p(9.17283818781952) q[8];
cx q[5],q[8];
cx q[6],q[8];
p(9.17283818781952) q[8];
cx q[6],q[8];
cx q[7],q[8];
p(9.17283818781952) q[8];
cx q[7],q[8];
cx q[9],q[22];
p(9.17283818781952) q[22];
cx q[9],q[22];
cx q[10],q[22];
p(9.17283818781952) q[22];
cx q[10],q[22];
u2(2.0,-pi) q[10];
cx q[11],q[22];
p(9.17283818781952) q[22];
cx q[11],q[22];
u2(2.0,-pi) q[11];
cx q[12],q[22];
p(9.17283818781952) q[22];
cx q[12],q[22];
u2(2.0,-pi) q[12];
cx q[13],q[22];
p(9.17283818781952) q[22];
cx q[13],q[22];
u2(2.0,-pi) q[13];
cx q[14],q[22];
p(9.17283818781952) q[22];
cx q[14],q[22];
u2(2.0,-pi) q[14];
cx q[15],q[22];
p(9.17283818781952) q[22];
cx q[15],q[22];
u2(2.0,-pi) q[15];
cx q[16],q[22];
p(9.17283818781952) q[22];
cx q[16],q[22];
u2(2.0,-pi) q[16];
cx q[17],q[22];
p(9.17283818781952) q[22];
cx q[17],q[22];
u2(2.0,-pi) q[17];
cx q[18],q[22];
p(9.17283818781952) q[22];
cx q[18],q[22];
u2(2.0,-pi) q[18];
cx q[19],q[22];
p(9.17283818781952) q[22];
cx q[19],q[22];
u2(2.0,-pi) q[19];
cx q[20],q[22];
p(9.17283818781952) q[22];
cx q[20],q[22];
u2(2.0,-pi) q[20];
cx q[21],q[22];
p(9.17283818781952) q[22];
cx q[21],q[22];
u2(2.0,-pi) q[21];
u2(2.0,-pi) q[22];
u2(2.0,-pi) q[9];
cx q[0],q[9];
p(9.17283818781952) q[9];
cx q[0],q[9];
cx q[0],q[10];
cx q[1],q[9];
p(9.17283818781952) q[10];
cx q[0],q[10];
cx q[0],q[11];
p(9.17283818781952) q[11];
cx q[0],q[11];
cx q[0],q[12];
p(9.17283818781952) q[12];
cx q[0],q[12];
cx q[0],q[13];
p(9.17283818781952) q[13];
cx q[0],q[13];
cx q[0],q[14];
p(9.17283818781952) q[14];
cx q[0],q[14];
cx q[0],q[15];
p(9.17283818781952) q[15];
cx q[0],q[15];
cx q[0],q[16];
p(9.17283818781952) q[16];
cx q[0],q[16];
cx q[0],q[17];
p(9.17283818781952) q[17];
cx q[0],q[17];
cx q[0],q[18];
p(9.17283818781952) q[18];
cx q[0],q[18];
cx q[0],q[19];
p(9.17283818781952) q[19];
cx q[0],q[19];
cx q[0],q[20];
p(9.17283818781952) q[20];
cx q[0],q[20];
cx q[0],q[21];
p(9.17283818781952) q[21];
cx q[0],q[21];
cx q[0],q[22];
p(9.17283818781952) q[22];
cx q[0],q[22];
ry(0.44425274861157) q[0];
p(9.17283818781952) q[9];
cx q[1],q[9];
cx q[1],q[10];
p(9.17283818781952) q[10];
cx q[1],q[10];
cx q[1],q[11];
p(9.17283818781952) q[11];
cx q[1],q[11];
cx q[1],q[12];
p(9.17283818781952) q[12];
cx q[1],q[12];
cx q[1],q[13];
p(9.17283818781952) q[13];
cx q[1],q[13];
cx q[1],q[14];
p(9.17283818781952) q[14];
cx q[1],q[14];
cx q[1],q[15];
p(9.17283818781952) q[15];
cx q[1],q[15];
cx q[1],q[16];
p(9.17283818781952) q[16];
cx q[1],q[16];
cx q[1],q[17];
p(9.17283818781952) q[17];
cx q[1],q[17];
cx q[1],q[18];
p(9.17283818781952) q[18];
cx q[1],q[18];
cx q[1],q[19];
p(9.17283818781952) q[19];
cx q[1],q[19];
cx q[1],q[20];
cx q[2],q[9];
p(9.17283818781952) q[20];
cx q[1],q[20];
cx q[1],q[21];
p(9.17283818781952) q[21];
cx q[1],q[21];
cx q[1],q[22];
p(9.17283818781952) q[22];
cx q[1],q[22];
ry(0.193886543235698) q[1];
p(9.17283818781952) q[9];
cx q[2],q[9];
cx q[2],q[10];
p(9.17283818781952) q[10];
cx q[2],q[10];
cx q[2],q[11];
p(9.17283818781952) q[11];
cx q[2],q[11];
cx q[2],q[12];
p(9.17283818781952) q[12];
cx q[2],q[12];
cx q[2],q[13];
p(9.17283818781952) q[13];
cx q[2],q[13];
cx q[2],q[14];
p(9.17283818781952) q[14];
cx q[2],q[14];
cx q[2],q[15];
p(9.17283818781952) q[15];
cx q[2],q[15];
cx q[2],q[16];
p(9.17283818781952) q[16];
cx q[2],q[16];
cx q[2],q[17];
p(9.17283818781952) q[17];
cx q[2],q[17];
cx q[2],q[18];
p(9.17283818781952) q[18];
cx q[2],q[18];
cx q[2],q[19];
p(9.17283818781952) q[19];
cx q[2],q[19];
cx q[2],q[20];
p(9.17283818781952) q[20];
cx q[2],q[20];
cx q[2],q[21];
p(9.17283818781952) q[21];
cx q[2],q[21];
cx q[2],q[22];
p(9.17283818781952) q[22];
cx q[2],q[22];
ry(0.443077189470742) q[2];
cx q[3],q[9];
p(9.17283818781952) q[9];
cx q[3],q[9];
cx q[3],q[10];
p(9.17283818781952) q[10];
cx q[3],q[10];
cx q[3],q[11];
p(9.17283818781952) q[11];
cx q[3],q[11];
cx q[3],q[12];
p(9.17283818781952) q[12];
cx q[3],q[12];
cx q[3],q[13];
p(9.17283818781952) q[13];
cx q[3],q[13];
cx q[3],q[14];
p(9.17283818781952) q[14];
cx q[3],q[14];
cx q[3],q[15];
p(9.17283818781952) q[15];
cx q[3],q[15];
cx q[3],q[16];
p(9.17283818781952) q[16];
cx q[3],q[16];
cx q[3],q[17];
p(9.17283818781952) q[17];
cx q[3],q[17];
cx q[3],q[18];
p(9.17283818781952) q[18];
cx q[3],q[18];
cx q[3],q[19];
p(9.17283818781952) q[19];
cx q[3],q[19];
cx q[3],q[20];
p(9.17283818781952) q[20];
cx q[3],q[20];
cx q[3],q[21];
p(9.17283818781952) q[21];
cx q[3],q[21];
cx q[3],q[22];
p(9.17283818781952) q[22];
cx q[3],q[22];
ry(0.308732445015847) q[3];
cx q[4],q[9];
p(9.17283818781952) q[9];
cx q[4],q[9];
cx q[4],q[10];
p(9.17283818781952) q[10];
cx q[4],q[10];
cx q[4],q[11];
p(9.17283818781952) q[11];
cx q[4],q[11];
cx q[4],q[12];
p(9.17283818781952) q[12];
cx q[4],q[12];
cx q[4],q[13];
p(9.17283818781952) q[13];
cx q[4],q[13];
cx q[4],q[14];
p(9.17283818781952) q[14];
cx q[4],q[14];
cx q[4],q[15];
p(9.17283818781952) q[15];
cx q[4],q[15];
cx q[4],q[16];
p(9.17283818781952) q[16];
cx q[4],q[16];
cx q[4],q[17];
p(9.17283818781952) q[17];
cx q[4],q[17];
cx q[4],q[18];
p(9.17283818781952) q[18];
cx q[4],q[18];
cx q[4],q[19];
p(9.17283818781952) q[19];
cx q[4],q[19];
cx q[4],q[20];
p(9.17283818781952) q[20];
cx q[4],q[20];
cx q[4],q[21];
p(9.17283818781952) q[21];
cx q[4],q[21];
cx q[4],q[22];
p(9.17283818781952) q[22];
cx q[4],q[22];
ry(0.724949174656261) q[4];
cx q[5],q[9];
p(9.17283818781952) q[9];
cx q[5],q[9];
cx q[5],q[10];
p(9.17283818781952) q[10];
cx q[5],q[10];
cx q[5],q[11];
p(9.17283818781952) q[11];
cx q[5],q[11];
cx q[5],q[12];
p(9.17283818781952) q[12];
cx q[5],q[12];
cx q[5],q[13];
p(9.17283818781952) q[13];
cx q[5],q[13];
cx q[5],q[14];
p(9.17283818781952) q[14];
cx q[5],q[14];
cx q[5],q[15];
p(9.17283818781952) q[15];
cx q[5],q[15];
cx q[5],q[16];
p(9.17283818781952) q[16];
cx q[5],q[16];
cx q[5],q[17];
p(9.17283818781952) q[17];
cx q[5],q[17];
cx q[5],q[18];
p(9.17283818781952) q[18];
cx q[5],q[18];
cx q[5],q[19];
p(9.17283818781952) q[19];
cx q[5],q[19];
cx q[5],q[20];
p(9.17283818781952) q[20];
cx q[5],q[20];
cx q[5],q[21];
p(9.17283818781952) q[21];
cx q[5],q[21];
cx q[5],q[22];
p(9.17283818781952) q[22];
cx q[5],q[22];
ry(0.186270323402387) q[5];
cx q[6],q[9];
p(9.17283818781952) q[9];
cx q[6],q[9];
cx q[6],q[10];
p(9.17283818781952) q[10];
cx q[6],q[10];
cx q[6],q[11];
p(9.17283818781952) q[11];
cx q[6],q[11];
cx q[6],q[12];
p(9.17283818781952) q[12];
cx q[6],q[12];
cx q[6],q[13];
p(9.17283818781952) q[13];
cx q[6],q[13];
cx q[6],q[14];
p(9.17283818781952) q[14];
cx q[6],q[14];
cx q[6],q[15];
p(9.17283818781952) q[15];
cx q[6],q[15];
cx q[6],q[16];
p(9.17283818781952) q[16];
cx q[6],q[16];
cx q[6],q[17];
p(9.17283818781952) q[17];
cx q[6],q[17];
cx q[6],q[18];
p(9.17283818781952) q[18];
cx q[6],q[18];
cx q[6],q[19];
p(9.17283818781952) q[19];
cx q[6],q[19];
cx q[6],q[20];
p(9.17283818781952) q[20];
cx q[6],q[20];
cx q[6],q[21];
p(9.17283818781952) q[21];
cx q[6],q[21];
cx q[6],q[22];
p(9.17283818781952) q[22];
cx q[6],q[22];
ry(0.212924428101626) q[6];
cx q[7],q[9];
p(9.17283818781952) q[9];
cx q[7],q[9];
cx q[7],q[10];
p(9.17283818781952) q[10];
cx q[7],q[10];
cx q[7],q[11];
p(9.17283818781952) q[11];
cx q[7],q[11];
cx q[7],q[12];
p(9.17283818781952) q[12];
cx q[7],q[12];
cx q[7],q[13];
p(9.17283818781952) q[13];
cx q[7],q[13];
cx q[7],q[14];
p(9.17283818781952) q[14];
cx q[7],q[14];
cx q[7],q[15];
p(9.17283818781952) q[15];
cx q[7],q[15];
cx q[7],q[16];
p(9.17283818781952) q[16];
cx q[7],q[16];
cx q[7],q[17];
p(9.17283818781952) q[17];
cx q[7],q[17];
cx q[7],q[18];
p(9.17283818781952) q[18];
cx q[7],q[18];
cx q[7],q[19];
p(9.17283818781952) q[19];
cx q[7],q[19];
cx q[7],q[20];
p(9.17283818781952) q[20];
cx q[7],q[20];
cx q[7],q[21];
p(9.17283818781952) q[21];
cx q[7],q[21];
cx q[7],q[22];
p(9.17283818781952) q[22];
cx q[7],q[22];
ry(0.629578496567906) q[7];
cx q[8],q[9];
p(9.17283818781952) q[9];
cx q[8],q[9];
cx q[8],q[10];
p(9.17283818781952) q[10];
cx q[8],q[10];
cx q[8],q[11];
p(9.17283818781952) q[11];
cx q[8],q[11];
cx q[8],q[12];
p(9.17283818781952) q[12];
cx q[8],q[12];
cx q[8],q[13];
p(9.17283818781952) q[13];
cx q[8],q[13];
cx q[8],q[14];
p(9.17283818781952) q[14];
cx q[8],q[14];
cx q[8],q[15];
p(9.17283818781952) q[15];
cx q[8],q[15];
cx q[8],q[16];
p(9.17283818781952) q[16];
cx q[8],q[16];
cx q[8],q[17];
p(9.17283818781952) q[17];
cx q[8],q[17];
cx q[8],q[18];
p(9.17283818781952) q[18];
cx q[8],q[18];
cx q[8],q[19];
p(9.17283818781952) q[19];
cx q[8],q[19];
cx q[8],q[20];
p(9.17283818781952) q[20];
cx q[8],q[20];
cx q[8],q[21];
p(9.17283818781952) q[21];
cx q[8],q[21];
cx q[8],q[22];
p(9.17283818781952) q[22];
cx q[8],q[22];
ry(0.584867467286787) q[8];
cx q[9],q[10];
p(9.17283818781952) q[10];
cx q[9],q[10];
cx q[9],q[11];
p(9.17283818781952) q[11];
cx q[9],q[11];
cx q[10],q[11];
p(9.17283818781952) q[11];
cx q[10],q[11];
cx q[9],q[12];
p(9.17283818781952) q[12];
cx q[9],q[12];
cx q[10],q[12];
p(9.17283818781952) q[12];
cx q[10],q[12];
cx q[11],q[12];
p(9.17283818781952) q[12];
cx q[11],q[12];
cx q[9],q[13];
p(9.17283818781952) q[13];
cx q[9],q[13];
cx q[10],q[13];
p(9.17283818781952) q[13];
cx q[10],q[13];
cx q[11],q[13];
p(9.17283818781952) q[13];
cx q[11],q[13];
cx q[12],q[13];
p(9.17283818781952) q[13];
cx q[12],q[13];
cx q[9],q[14];
p(9.17283818781952) q[14];
cx q[9],q[14];
cx q[10],q[14];
p(9.17283818781952) q[14];
cx q[10],q[14];
cx q[11],q[14];
p(9.17283818781952) q[14];
cx q[11],q[14];
cx q[12],q[14];
p(9.17283818781952) q[14];
cx q[12],q[14];
cx q[13],q[14];
p(9.17283818781952) q[14];
cx q[13],q[14];
cx q[9],q[15];
p(9.17283818781952) q[15];
cx q[9],q[15];
cx q[10],q[15];
p(9.17283818781952) q[15];
cx q[10],q[15];
cx q[11],q[15];
p(9.17283818781952) q[15];
cx q[11],q[15];
cx q[12],q[15];
p(9.17283818781952) q[15];
cx q[12],q[15];
cx q[13],q[15];
p(9.17283818781952) q[15];
cx q[13],q[15];
cx q[14],q[15];
p(9.17283818781952) q[15];
cx q[14],q[15];
cx q[9],q[16];
p(9.17283818781952) q[16];
cx q[9],q[16];
cx q[10],q[16];
p(9.17283818781952) q[16];
cx q[10],q[16];
cx q[11],q[16];
p(9.17283818781952) q[16];
cx q[11],q[16];
cx q[12],q[16];
p(9.17283818781952) q[16];
cx q[12],q[16];
cx q[13],q[16];
p(9.17283818781952) q[16];
cx q[13],q[16];
cx q[14],q[16];
p(9.17283818781952) q[16];
cx q[14],q[16];
cx q[15],q[16];
p(9.17283818781952) q[16];
cx q[15],q[16];
cx q[9],q[17];
p(9.17283818781952) q[17];
cx q[9],q[17];
cx q[10],q[17];
p(9.17283818781952) q[17];
cx q[10],q[17];
cx q[11],q[17];
p(9.17283818781952) q[17];
cx q[11],q[17];
cx q[12],q[17];
p(9.17283818781952) q[17];
cx q[12],q[17];
cx q[13],q[17];
p(9.17283818781952) q[17];
cx q[13],q[17];
cx q[14],q[17];
p(9.17283818781952) q[17];
cx q[14],q[17];
cx q[15],q[17];
p(9.17283818781952) q[17];
cx q[15],q[17];
cx q[16],q[17];
p(9.17283818781952) q[17];
cx q[16],q[17];
cx q[9],q[18];
p(9.17283818781952) q[18];
cx q[9],q[18];
cx q[10],q[18];
p(9.17283818781952) q[18];
cx q[10],q[18];
cx q[11],q[18];
p(9.17283818781952) q[18];
cx q[11],q[18];
cx q[12],q[18];
p(9.17283818781952) q[18];
cx q[12],q[18];
cx q[13],q[18];
p(9.17283818781952) q[18];
cx q[13],q[18];
cx q[14],q[18];
p(9.17283818781952) q[18];
cx q[14],q[18];
cx q[15],q[18];
p(9.17283818781952) q[18];
cx q[15],q[18];
cx q[16],q[18];
p(9.17283818781952) q[18];
cx q[16],q[18];
cx q[17],q[18];
p(9.17283818781952) q[18];
cx q[17],q[18];
cx q[9],q[19];
p(9.17283818781952) q[19];
cx q[9],q[19];
cx q[10],q[19];
p(9.17283818781952) q[19];
cx q[10],q[19];
cx q[11],q[19];
p(9.17283818781952) q[19];
cx q[11],q[19];
cx q[12],q[19];
p(9.17283818781952) q[19];
cx q[12],q[19];
cx q[13],q[19];
p(9.17283818781952) q[19];
cx q[13],q[19];
cx q[14],q[19];
p(9.17283818781952) q[19];
cx q[14],q[19];
cx q[15],q[19];
p(9.17283818781952) q[19];
cx q[15],q[19];
cx q[16],q[19];
p(9.17283818781952) q[19];
cx q[16],q[19];
cx q[17],q[19];
p(9.17283818781952) q[19];
cx q[17],q[19];
cx q[18],q[19];
p(9.17283818781952) q[19];
cx q[18],q[19];
cx q[9],q[20];
p(9.17283818781952) q[20];
cx q[9],q[20];
cx q[10],q[20];
p(9.17283818781952) q[20];
cx q[10],q[20];
cx q[11],q[20];
p(9.17283818781952) q[20];
cx q[11],q[20];
cx q[12],q[20];
p(9.17283818781952) q[20];
cx q[12],q[20];
cx q[13],q[20];
p(9.17283818781952) q[20];
cx q[13],q[20];
cx q[14],q[20];
p(9.17283818781952) q[20];
cx q[14],q[20];
cx q[15],q[20];
p(9.17283818781952) q[20];
cx q[15],q[20];
cx q[16],q[20];
p(9.17283818781952) q[20];
cx q[16],q[20];
cx q[17],q[20];
p(9.17283818781952) q[20];
cx q[17],q[20];
cx q[18],q[20];
p(9.17283818781952) q[20];
cx q[18],q[20];
cx q[19],q[20];
p(9.17283818781952) q[20];
cx q[19],q[20];
cx q[9],q[21];
p(9.17283818781952) q[21];
cx q[9],q[21];
cx q[10],q[21];
p(9.17283818781952) q[21];
cx q[10],q[21];
cx q[11],q[21];
p(9.17283818781952) q[21];
cx q[11],q[21];
cx q[12],q[21];
p(9.17283818781952) q[21];
cx q[12],q[21];
cx q[13],q[21];
p(9.17283818781952) q[21];
cx q[13],q[21];
cx q[14],q[21];
p(9.17283818781952) q[21];
cx q[14],q[21];
cx q[15],q[21];
p(9.17283818781952) q[21];
cx q[15],q[21];
cx q[16],q[21];
p(9.17283818781952) q[21];
cx q[16],q[21];
cx q[17],q[21];
p(9.17283818781952) q[21];
cx q[17],q[21];
cx q[18],q[21];
p(9.17283818781952) q[21];
cx q[18],q[21];
cx q[19],q[21];
p(9.17283818781952) q[21];
cx q[19],q[21];
cx q[20],q[21];
p(9.17283818781952) q[21];
cx q[20],q[21];
cx q[9],q[22];
p(9.17283818781952) q[22];
cx q[9],q[22];
cx q[10],q[22];
p(9.17283818781952) q[22];
cx q[10],q[22];
ry(0.804734460140074) q[10];
cx q[11],q[22];
p(9.17283818781952) q[22];
cx q[11],q[22];
ry(0.0523727252283163) q[11];
cx q[12],q[22];
p(9.17283818781952) q[22];
cx q[12],q[22];
ry(0.264774891767032) q[12];
cx q[13],q[22];
p(9.17283818781952) q[22];
cx q[13],q[22];
ry(0.0229737383267014) q[13];
cx q[14],q[22];
p(9.17283818781952) q[22];
cx q[14],q[22];
ry(0.871516425450299) q[14];
cx q[15],q[22];
p(9.17283818781952) q[22];
cx q[15],q[22];
ry(0.694273406174942) q[15];
cx q[16],q[22];
p(9.17283818781952) q[22];
cx q[16],q[22];
ry(0.0656419610304003) q[16];
cx q[17],q[22];
p(9.17283818781952) q[22];
cx q[17],q[22];
ry(0.210681906604621) q[17];
cx q[18],q[22];
p(9.17283818781952) q[22];
cx q[18],q[22];
ry(0.550660834026107) q[18];
cx q[19],q[22];
p(9.17283818781952) q[22];
cx q[19],q[22];
ry(0.255210338470069) q[19];
cx q[20],q[22];
p(9.17283818781952) q[22];
cx q[20],q[22];
ry(0.256482609387276) q[20];
cx q[21],q[22];
p(9.17283818781952) q[22];
cx q[21],q[22];
ry(0.307314893198253) q[21];
ry(0.775408933301947) q[22];
cx q[21],q[22];
cx q[20],q[21];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[18];
cx q[16],q[17];
cx q[15],q[16];
cx q[14],q[15];
cx q[13],q[14];
cx q[12],q[13];
cx q[11],q[12];
cx q[10],q[11];
ry(0.64663155072772) q[11];
ry(0.389602974347502) q[12];
ry(0.515326072288199) q[13];
ry(0.37071100771637) q[14];
ry(0.658477190203549) q[15];
ry(0.954412866916806) q[16];
ry(0.444782225528972) q[17];
ry(0.458577865853887) q[18];
ry(0.964352303067845) q[19];
ry(0.901970925343079) q[20];
ry(0.195478431872271) q[21];
ry(0.826398785765007) q[22];
ry(0.947425590050769) q[9];
cx q[9],q[10];
ry(0.708781081770354) q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
ry(0.301055019975721) q[0];
ry(0.156209909198447) q[1];
ry(0.348280682436003) q[2];
ry(0.784714702699245) q[3];
ry(0.479141579548548) q[4];
ry(0.349429476110932) q[5];
ry(0.12120779776252) q[6];
ry(0.103689232509187) q[7];
ry(0.964929842036502) q[8];
ry(0.391168946218891) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
measure q[16] -> meas[16];
measure q[17] -> meas[17];
measure q[18] -> meas[18];
measure q[19] -> meas[19];
measure q[20] -> meas[20];
measure q[21] -> meas[21];
measure q[22] -> meas[22];
