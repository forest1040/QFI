import math
import random
from typing import Callable, List

import numpy as np

from qulacs import Observable, ParametricQuantumCircuit, QuantumState, gate
from qulacs.state import inner_product


def python_backprop(circ: ParametricQuantumCircuit, obs: Observable) -> List[float]:
    def backprop_inner_product(
        circ: ParametricQuantumCircuit, bistate: QuantumState
    ) -> List[float]:
        n = circ.get_qubit_count()
        state = QuantumState(n)
        state.set_zero_state()
        circ.update_quantum_state(state)

        num_gates = circ.get_gate_count()
        inverse_parametric_gate_position = [-1] * num_gates
        for i in range(circ.get_parameter_count()):
            inverse_parametric_gate_position[circ.get_parametric_gate_position(i)] = i
        #ans = [0.0] * circ.get_parameter_count()
        param_gates = circ.get_parameter_count()
        qfim = np.zeros((param_gates, param_gates))

        astate = QuantumState(n)
        for i in range(param_gates - 1, -1, -1):
            gate_now = circ.get_gate(i)
            if inverse_parametric_gate_position[i] != -1:
                astate.load(state)
                if gate_now.get_name() == "ParametricRX":
                    rcpi = gate.RX(gate_now.get_target_index_list()[0], math.pi)
                elif gate_now.get_name() == "ParametricRY":
                    rcpi = gate.RY(gate_now.get_target_index_list()[0], math.pi)
                elif gate_now.get_name() == "ParametricRZ":
                    rcpi = gate.RZ(gate_now.get_target_index_list()[0], math.pi)
                else:
                    raise RuntimeError()
                rcpi.update_quantum_state(astate)
                rcpi.update_quantum_state(state)
                # ans[inverse_parametric_gate_position[i]] = (
                #     inner_product(bistate, astate).real / 2.0
                # )
            agate = gate_now.get_inverse()
            #agate.update_quantum_state(bistate)
            agate.update_quantum_state(state)

            state2 = QuantumState(n)
            state2.set_zero_state()
            circ.update_quantum_state(state2)
            psi = QuantumState(n)
            for j in range(param_gates - 1, -1, -1):
                gate_now = circ.get_gate(i)
                if inverse_parametric_gate_position[i] != -1:
                    psi.load(state2)
                    if gate_now.get_name() == "ParametricRX":
                        rcpi = gate.RX(gate_now.get_target_index_list()[0], math.pi)
                    elif gate_now.get_name() == "ParametricRY":
                        rcpi = gate.RY(gate_now.get_target_index_list()[0], math.pi)
                    elif gate_now.get_name() == "ParametricRZ":
                        rcpi = gate.RZ(gate_now.get_target_index_list()[0], math.pi)
                    else:
                        raise RuntimeError()
                    rcpi.update_quantum_state(psi)
                    rcpi.update_quantum_state(state2)
                    # ans[inverse_parametric_gate_position[i]] = (
                    #     inner_product(bistate, astate).real / 2.0
                    # )
                g = gate_now.get_inverse()
                #psi.update_quantum_state(bistate)
                g.update_quantum_state(state2)

                qfim[i][j] += (
                    #abs(inner_product(psi_shift, psi)) ** 2 * sign_i * sign_j * -1
                    #abs(inner_product(astate, psi)) ** 2 * -1
                    abs(inner_product(state, state2)) ** 2
                )
                # if i != j:
                #     # The QFIM is symmetric
                #     qfim[j][i] = qfim[i][j]

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

#qf = QuantumFisher(circuit)
#result = qf.get_qfisher_matrix()

obs = Observable(n_qubit)
for i in range(n_qubit):
    obs.add_operator(1.0, f"Z {i}")

result = python_backprop(circuit, obs)

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
