import numpy as np
from qulacs import (
    Observable,
    ParametricQuantumCircuit,
    QuantumState,
)
from qulacs.state import inner_product

from fisher import QuantumFisher

# from qulacsvis import circuit_drawer

from backprop import python_backprop


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

obs = Observable(n_qubit)
for i in range(n_qubit):
    obs.add_operator(1.0, f"Z {i}")

# grad = circuit.backprop(obs)
# print(grad)

grad, bistates, astates = python_backprop(circuit, obs)
# print(grad)
# print(bistates)
# print(astates)

n = circuit.get_qubit_count()
state = QuantumState(n)
state.set_zero_state()
circuit.update_quantum_state(state)
bistate = QuantumState(n)
astate = QuantumState(n)
obs.apply_to_state(astate, state, bistate)
# bistate.multiply_coef(2)

num_param = len(grad)
qfim = np.zeros((num_param, num_param))
for k in range(num_param):
    for l in range(num_param):
        dk = bistates[k]
        dl = bistates[l]
        ak = astates[k]
        al = astates[l]
        # print(f"inner_product(dk, dl): {inner_product(dk, dl)}")
        # print(f"inner_product(dk, st): {inner_product(dk, state)}")
        # print(f"inner_product(st, dl): {inner_product(state, dl)}")
        # qfim[k][l] = (
        #     4
        #     * (
        #         inner_product(ak, al)
        #         - inner_product(ak, state) * inner_product(state, al)
        #     ).real
        # )
        # qfim[k][l] = (
        #     4
        #     * (
        #         abs(inner_product(ak, al)) ** 2
        #         - abs(inner_product(ak, state)) ** 2
        #         * abs(inner_product(state, al)) ** 2
        #     ).real
        # )
        qfim[k][l] = 4 * (
            inner_product(ak, al).real
            - inner_product(ak, dl).real * inner_product(dk, al).real
        )

        # inner_product(dk, dl)
        # - inner_product(dk, state) * inner_product(state, dl)

# print(qfim)

for i in range(num_param):
    tmp = ""
    for j in range(num_param):
        tmp += str("{:.08f}, ".format(qfim[i][j]))
    print(tmp)

    # qfim[k][l] = (
    #     4
    #     * (
    #         abs(inner_product(dk, dl)) ** 2
    #         - abs(inner_product(dk, state)) ** 2
    #         * abs(inner_product(state, dl)) ** 2
    #     ).real
    # )
