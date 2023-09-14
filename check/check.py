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


# def _make_fullgate(list_SiteAndOperator, n_qubit):
#     I_mat = np.eye(2, dtype=complex)
#     list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
#     list_SingleGates = []
#     cnt = 0
#     for i in range(n_qubit):
#         if i in list_Site:
#             list_SingleGates.append(list_SiteAndOperator[cnt][1])
#             cnt += 1
#         else:
#             list_SingleGates.append(I_mat)
#     return reduce(np.kron, list_SingleGates)


# def _make_hamiltonian(
#     n_qubit, rng: Optional[Generator] = None, seed: Optional[int] = 0
# ):
#     if rng is None:
#         rng = default_rng(seed)
#     X_mat = np.array([[0, 1], [1, 0]])
#     Z_mat = np.array([[1, 0], [0, -1]])
#     ham = np.zeros((2**n_qubit, 2**n_qubit), dtype=complex)
#     for i in range(n_qubit):
#         Jx = rng.uniform(-1.0, 1.0)
#         ham += Jx * _make_fullgate([[i, X_mat]], n_qubit)
#         for j in range(i + 1, n_qubit):
#             J_ij = rng.uniform(-1.0, 1.0)
#             ham += J_ij * _make_fullgate([[i, Z_mat], [j, Z_mat]], n_qubit)
#     return ham


# def _create_time_evol_gate(
#     n_qubit, time_step=0.77, rng: Optional[Generator] = None, seed: Optional[int] = 0
# ):
#     if rng is None:
#         rng = default_rng(seed)

#     ham = _make_hamiltonian(n_qubit, rng)
#     # Create time evolution operator by diagonalization.
#     # H*P = P*D <-> H = P*D*P^dagger
#     diag, eigen_vecs = np.linalg.eigh(ham)
#     time_evol_op = np.dot(
#         np.dot(eigen_vecs, np.diag(np.exp(-1j * time_step * diag))), eigen_vecs.T.conj()
#     )  # e^-iHT

#     # Convert to a qulacs gate
#     time_evol_gate = DenseMatrix([i for i in range(n_qubit)], time_evol_op)

#     return time_evol_gate


# def _create_time_evol_gate(n_qubit, time_step):
#     matrix = [
#         [
#             8.62702577e-01 + 0.2195573j,
#             2.27648462e-16 + 0.43498929j,
#             -5.60385725e-18 - 0.12127137j,
#             5.99535809e-02 + 0.00468961j,
#         ],
#         [
#             3.21787748e-16 + 0.43498929j,
#             8.62702577e-01 - 0.2195573j,
#             5.99535809e-02 - 0.00468961j,
#             -4.06942584e-17 - 0.12127137j,
#         ],
#         [
#             -1.64764166e-17 - 0.12127137j,
#             5.99535809e-02 - 0.00468961j,
#             8.62702577e-01 - 0.2195573j,
#             5.12539044e-16 + 0.43498929j,
#         ],
#         [
#             5.99535809e-02 + 0.00468961j,
#             -4.25925761e-17 - 0.12127137j,
#             5.97704088e-16 + 0.43498929j,
#             8.62702577e-01 + 0.2195573j,
#         ],
#     ]
#     return DenseMatrix([0, 1], matrix)


def create_qcl_ansatz(
    n_qubit: int, c_depth: int, time_step: float = 0.5, seed: Optional[int] = 0
) -> ParametricQuantumCircuit:
    circuit = ParametricQuantumCircuit(n_qubit)
    # time_evol_gate = _create_time_evol_gate(n_qubit, time_step)
    for _ in range(c_depth):
        # circuit.add_gate(time_evol_gate)
        for i in range(n_qubit):
            circuit.add_parametric_RX_gate(i, 0)
            circuit.add_parametric_RZ_gate(i, 0)
            circuit.add_parametric_RX_gate(i, 0)
    return circuit


n_qubit = 2
depth = 1
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


for i in range(n_qubit):
    circuit.add_RY_gate(i, np.arcsin(x) * 2)
    circuit.add_RZ_gate(i, np.arccos(x * x) * 2)

ansatz = create_qcl_ansatz(n_qubit, depth)
for i in range(len(theta)):
    ansatz.set_parameter(i, theta[i])

circuit.merge_circuit(ansatz)
qf = QuantumFisher(circuit)
result = qf.get_qfisher_matrix()

print("x:", x)
print("RY:", np.arcsin(x) * 2)
print("RZ:", np.arccos(x * x) * 2)
print("theta:", theta)
print("QFI:", result)

circuit_drawer(circuit, output_method="text")
