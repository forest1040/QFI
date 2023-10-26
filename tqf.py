from typing import List
import math
import numpy as np
from qulacs import (
    Observable,
    QuantumCircuit,
    ParametricQuantumCircuit,
    QuantumState,
    gate,
)
from qulacs.state import inner_product

# from fisher import QuantumFisher
from bp import python_backprop


def get_differential_gate(g):
    if g.get_name() == "ParametricRX":
        rcpi = gate.RX(g.get_target_index_list()[0], math.pi)
    elif g.get_name() == "ParametricRY":
        rcpi = gate.RY(g.get_target_index_list()[0], math.pi)
    elif g.get_name() == "ParametricRZ":
        rcpi = gate.RZ(g.get_target_index_list()[0], math.pi)
    else:
        raise RuntimeError()
    return rcpi


def fisher(input_circuit: QuantumCircuit, ansatz: ParametricQuantumCircuit):
    n = input_circuit.get_qubit_count()
    chi = QuantumState(n)
    input_circuit.update_quantum_state(chi)
    phi = chi.copy()
    gate = ansatz.get_gate(0)
    gate.update_quantum_state(chi)
    psi = chi.copy()
    rcpi = get_differential_gate(gate)
    rcpi.update_quantum_state(phi)

    num_param = ansatz.get_gate_count()
    print(f"num_param: {num_param}")

    T = np.zeros(num_param, dtype=complex)
    T[0] = inner_product(chi, phi)
    L = np.zeros((num_param, num_param), dtype=complex)
    L[0][0] = inner_product(phi, phi)

    for j in range(1, num_param):
        lambda_state = psi.copy()
        phi = psi.copy()
        gate = ansatz.get_gate(j)
        gate.update_quantum_state(phi)
        L[j][j] = inner_product(phi, phi)
        print(f"j:{j}")
        for i in range(j - 1, 0, -1):
            print(f"i:{i}")
            gate = ansatz.get_gate(i + 1).get_inverse()
            gate.update_quantum_state(phi)
            gate = ansatz.get_gate(i).get_inverse()
            gate.update_quantum_state(lambda_state)
            myu = lambda_state.copy()
            rcpi = get_differential_gate(gate)
            rcpi.update_quantum_state(myu)
            L[i][j] = inner_product(myu, phi)
        T[j] = inner_product(chi, phi)
        gate = ansatz.get_gate(j).get_inverse()
        gate.update_quantum_state(psi)

    print(f"T: {T}")
    print(f"L: {L}")

    qfi = np.zeros((num_param, num_param))
    for i in range(num_param):
        for j in range(num_param):
            if i <= j:
                # print(f"T[i]: {T[i]}")
                # print(f"T[i].conj(): {T[i].conj()}")
                qfi[i][j] = L[i][j] - T[i].conj() * T[j]
                # qfi[i][j] = L[i][j] - T[i] * T[j]
            else:
                qfi[i][j] = L[j][i].conj() - T[i].conj() * T[j]
                # qfi[i][j] = L[j][i].conj() - T[i] * T[j]
    return qfi


def main():
    n_qubit = 2
    x = 0.27392337

    theta = [
        4.002148315014479,
        1.6951199159934145,
        0.25744424357926954,
        0.10384619671527331,
        5.109927617709579,
        5.735012432197602,
    ]

    input_circuit = QuantumCircuit(n_qubit)

    for i in range(n_qubit):
        input_circuit.add_RY_gate(i, np.arcsin(x) * 2)
        input_circuit.add_RZ_gate(i, np.arccos(x * x) * 2)

    ansatz = ParametricQuantumCircuit(n_qubit)
    ansatz.add_parametric_RX_gate(0, theta[0])
    ansatz.add_parametric_RZ_gate(0, theta[1])
    ansatz.add_parametric_RX_gate(0, theta[2])
    ansatz.add_parametric_RX_gate(1, theta[3])
    ansatz.add_parametric_RZ_gate(1, theta[4])
    ansatz.add_parametric_RX_gate(1, theta[5])

    print("theta:", theta)
    print("QFI")
    qfi = fisher(input_circuit, ansatz)
    row_size, col_size = qfi.shape
    for i in range(row_size):
        tmp = ""
        for j in range(col_size):
            tmp += str("{:.08f}, ".format(qfi[i][j]))
        print(tmp)


main()
