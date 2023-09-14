from typing import Callable, List, Optional, Tuple
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from functools import reduce

import numpy as np
from qulacs import (
    Observable,
    ParametricQuantumCircuit,
    QuantumState,
)
from qulacs.gate import DenseMatrix

from fisher import QuantumFisher

from qulacsvis import circuit_drawer


n_qubit = 2
circuit = ParametricQuantumCircuit(n_qubit)
# x = 0.27392337

theta = [
    4.002148315014479,
    1.6951199159934145,
    0.25744424357926954,
    0.10384619671527331,
    5.109927617709579,
    5.735012432197602,
]

circuit = ParametricQuantumCircuit(n_qubit)

# for i in range(n_qubit):
#     circuit.add_RY_gate(i, np.arcsin(x) * 2)
#     circuit.add_RZ_gate(i, np.arccos(x * x) * 2)

circuit.add_parametric_RX_gate(0, theta[0])
circuit.add_parametric_RZ_gate(0, theta[1])
circuit.add_parametric_RX_gate(0, theta[2])
circuit.add_parametric_RX_gate(1, theta[3])
circuit.add_parametric_RZ_gate(1, theta[4])
circuit.add_parametric_RX_gate(1, theta[5])

qf = QuantumFisher(circuit)
result = qf.get_qfisher_matrix()

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

circuit_drawer(circuit, output_method="text")
