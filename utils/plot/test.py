

rl = [13506,	19726,	17127,	2615,	6145,	6963,	24891]
qiskit = [15323,	20716,	19247,	4111,	6825,	7383,	25667]

for i in range(len(rl)):
    print(round(1 - rl[i]/qiskit[i],4))
