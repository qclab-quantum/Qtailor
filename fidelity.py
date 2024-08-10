import math
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity

from utils.circuit_util import CircutUtil
#others: 'measure','barrier',
gates_1=['u1', 'u2', 'u3', 'id','p','ry','h']
gates_2=['cp','cx','swap',]
basis_gates=gates_1+gates_2
simulator = AerSimulator()

from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)

def get_noise_model():
    noise_model = NoiseModel()
    # Define a depolarizing error for the CX gate
    cx_error = depolarizing_error(0.5, num_qubits=2)  # p is the error probability

    noise_model.add_all_qubit_quantum_error(cx_error, ['cx'])

    #error = depolarizing_error(0.001, 1)
    #noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3','rz'])

    # T1 and T2 values for qubits 0-3
    T1s = np.random.normal(50e1, 10e1, 4)  # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(70e1, 10e1, 4)  # Sampled from normal distribution mean 50 microsec
    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)])
    time_id = 50  # (single X90 pulse)
    errors_u1 = [thermal_relaxation_error(t1, t2, time_id)
                 for t1, t2 in zip(T1s, T2s)]
    for j in range(4):
        noise_model.add_quantum_error(errors_u1[j], "id", [j])


def flip_noise_model():
    # Example error probabilities
    err_rate = 0.5
    err_pair = [('X',err_rate ), ('I', 1-err_rate)]

    # QuantumError objects
    error_gate1 = pauli_error(err_pair)
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    #noise_bit_flip.add_all_qubit_quantum_error(error_gate1, gates_1)
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ['swap'])
    return noise_bit_flip

t1=-1
t2=-1
def T_noise_model(t1=50000,t2=70000):
    n=10
    # T1 and T2 values for qubits 0-3
    T1s = np.random.normal(t1, t1*0.01, n)  # 50 e3=Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(t2, t2*0.01, n)

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(n)])

    # Instruction times (in nanoseconds)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000  # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                      for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
        thermal_relaxation_error(t1b, t2b, time_cx))
        for t1a, t2a in zip(T1s, T2s)]
        for t1b, t2b in zip(T1s, T2s)]
    # Add errors to noise model
    noise_thermal = NoiseModel()
    #noise_thermal.add_all_qubit_quantum_error(errors_cx[0][0], gates_2 )
    noise_thermal.add_all_qubit_quantum_error(errors_cx[0][0], ['swap'] )
    return noise_thermal


def circuit_fidelity_benchmark(circuit,coupling_map,type:str,t1,t2):
    noise_model=T_noise_model(t1,t2)
    sim_ideal = AerSimulator()
    sim_noise = AerSimulator(noise_model=noise_model)
    circuit_ideal = None
    circ_tnoise = None
    if type == 'qiskit':
        circuit_ideal = transpile(circuit, backend=sim_ideal,coupling_map=coupling_map,basis_gates=basis_gates)
        circ_tnoise = transpile(circuit, backend=sim_noise,coupling_map=coupling_map,basis_gates=basis_gates)
    elif type == "rl":
        initial_layout = list(range(len(circuit.qubits)))
        circuit_ideal = transpile(circuit, backend=sim_ideal,initial_layout=initial_layout,coupling_map=coupling_map,basis_gates=basis_gates)
        circ_tnoise = transpile(circuit, backend=sim_noise,initial_layout=initial_layout,coupling_map=coupling_map,basis_gates=basis_gates)


    # print(f"{type} circuit_ideal depth={circuit_ideal.depth()} opts = {dict(circuit_ideal.count_ops())}")
    # print(f"{type} circ_tnoise depth={circ_tnoise.depth()} opts = {dict(circ_tnoise.count_ops())}")

    result_ideal = sim_ideal.run(circuit_ideal).result()
    counts = result_ideal.get_counts(0)
    #print(counts)
    #plot_histogram(counts).show()


    result_noise = sim_noise.run(circ_tnoise).result()
    counts_noise = result_noise.get_counts(0)
    #print(counts_noise)
    #plot_histogram(counts_noise).show()

    fidelity = calc_fidelity(counts, counts_noise)
    return fidelity


#使用 Bhattacharyya 距离近似
def calc_fidelity(result1,result2):
    #合并 keys
    bit_strings = list(set(list(result1.keys()) + list(result2.keys())))
    cnt = 0

    for bit_string in bit_strings:
        if bit_string in result1:
            cnt += result1[bit_string]

    fidelity = 0
    for bit_string in bit_strings:
        if bit_string not in result1 or bit_string not in result2:
            continue
        probility1=  result1[bit_string]/cnt
        probility2=  result2[bit_string]/cnt
        #print(f"p1={probility1}, p2={probility2}")
        fidelity += math.sqrt(probility1*probility2)
    return fidelity

if __name__ == '__main__':
    f = []
    n_qubits=4
    circ = QuantumCircuit(n_qubits)
    circ.h(0)
    for qubit in range(n_qubits - 1):
        circ.cx(qubit, qubit + 1)
    circ.measure_all()
