OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[13];
cx q[13],q[12];
cx q[12],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[14],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[14];
cx q[14],q[12];
cx q[12],q[13];
cx q[13],q[11];
cx q[11],q[10];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[11];
cx q[11],q[12];
cx q[12],q[15];
cx q[15],q[11];
cx q[11],q[14];
cx q[14],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[13];
cx q[13],q[12];
cx q[12],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[12];
cx q[12],q[13];
