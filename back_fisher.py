import math
import random
from typing import Callable, List

import numpy as np

from qulacs import Observable, ParametricQuantumCircuit, QuantumState, gate
from qulacs.state import inner_product


def fisher(circ: ParametricQuantumCircuit, obs: Observable) -> List[float]:
    def backprop_inner_product(
        circ: ParametricQuantumCircuit, bistate: QuantumState
    ) -> List[float]:
        n = circ.get_qubit_count()
        state = QuantumState(n)
        state.set_zero_state()
        circ.update_quantum_state(state)

        bistate_org = bistate.copy()

        num_gates = circ.get_gate_count()
        inverse_parametric_gate_position = [-1] * num_gates
        for i in range(circ.get_parameter_count()):
            inverse_parametric_gate_position[circ.get_parametric_gate_position(i)] = i
        # ans = [0.0] * circ.get_parameter_count()
        param_gates = circ.get_parameter_count()
        qfim = np.zeros((param_gates, param_gates))

        # print("inverse_parametric_gate_position:", inverse_parametric_gate_position)

        state_k = QuantumState(n)
        k = 0
        for i in range(num_gates - 1, -1, -1):
            gate_now = circ.get_gate(i)
            if inverse_parametric_gate_position[i] != -1:
                state_k.load(state)
                if gate_now.get_name() == "ParametricRX":
                    rcpi = gate.RX(gate_now.get_target_index_list()[0], math.pi)
                elif gate_now.get_name() == "ParametricRY":
                    rcpi = gate.RY(gate_now.get_target_index_list()[0], math.pi)
                elif gate_now.get_name() == "ParametricRZ":
                    rcpi = gate.RZ(gate_now.get_target_index_list()[0], math.pi)
                else:
                    raise RuntimeError()
                rcpi.update_quantum_state(state_k)

                bistate2 = bistate_org.copy()
                state2 = QuantumState(n)
                state2.set_zero_state()
                circ.update_quantum_state(state2)
                state_l = QuantumState(n)
                l = 0
                for j in range(num_gates - 1, -1, -1):
                    gate_now = circ.get_gate(j)
                    if inverse_parametric_gate_position[j] != -1:
                        state_l.load(state2)
                        if gate_now.get_name() == "ParametricRX":
                            rcpi = gate.RX(gate_now.get_target_index_list()[0], math.pi)
                        elif gate_now.get_name() == "ParametricRY":
                            rcpi = gate.RY(gate_now.get_target_index_list()[0], math.pi)
                        elif gate_now.get_name() == "ParametricRZ":
                            rcpi = gate.RZ(gate_now.get_target_index_list()[0], math.pi)
                        else:
                            raise RuntimeError()
                        rcpi.update_quantum_state(state_l)

                        qfim[k][l] = (
                            4
                            * (
                                inner_product(state_k, state_l)
                                - (
                                    inner_product(state_k, state)
                                    * inner_product(state, state_l)
                                )
                            ).real
                        )
                        l += 1

                    agate = gate_now.get_inverse()
                    agate.update_quantum_state(bistate2)
                    # agate.update_quantum_state(state2)

                k += 1

            agate = gate_now.get_inverse()
            agate.update_quantum_state(bistate)
        #            agate.update_quantum_state(state)

        return qfim

    n = circ.get_qubit_count()
    state = QuantumState(n)
    state.set_zero_state()
    circ.update_quantum_state(state)
    bistate = QuantumState(n)
    astate = QuantumState(n)

    obs.apply_to_state(astate, state, bistate)
    bistate.multiply_coef(2)

    qfim = backprop_inner_product(circ, bistate)
    return qfim


n_qubit = 2
circuit = ParametricQuantumCircuit(n_qubit)
x = 0.27392337

theta = [
    4.002148315014479,
    1.6951199159934145,
    0.25744424357926954,
    0.10384619671527331,
    5.109927617709579,
    5.735012432197602,
]

circuit = ParametricQuantumCircuit(n_qubit)

for i in range(n_qubit):
    circuit.add_RY_gate(i, np.arcsin(x) * 2)
    circuit.add_RZ_gate(i, np.arccos(x * x) * 2)

circuit.add_parametric_RX_gate(0, theta[0])
circuit.add_parametric_RZ_gate(0, theta[1])
circuit.add_parametric_RX_gate(0, theta[2])
circuit.add_parametric_RX_gate(1, theta[3])
circuit.add_parametric_RZ_gate(1, theta[4])
circuit.add_parametric_RX_gate(1, theta[5])

# qf = QuantumFisher(circuit)
# result = qf.get_qfisher_matrix()

obs = Observable(n_qubit)
for i in range(n_qubit):
    obs.add_operator(1.0, f"Z {i}")

result = fisher(circuit, obs)

# print("x:", x)
# print("RY:", np.arcsin(x) * 2)
# print("RZ:", np.arccos(x * x) * 2)
print("theta:", theta)
# print("QFI:", result)
# print(result.shape)
print("QFI")
row_size, col_size = result.shape
for i in range(row_size):
    tmp = ""
    for j in range(col_size):
        tmp += str("{:.08f}, ".format(result[i][j]))
    print(tmp)
