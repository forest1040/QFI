# The following were used as references.
# https://github.com/PaddlePaddle/Quantum/blob/master/paddle_quantum/fisher.py

import numpy as np

from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.state import inner_product


class QuantumFisher:
    def __init__(self, circuit: ParametricQuantumCircuit):
        self.circuit = circuit

    def update_param(self, params):
        for i in range(len(params)):
            self.circuit.set_parameter(i, params[i])

    def get_qfisher_matrix(self) -> np.ndarray:
        list_param = []
        for i in range(self.circuit.get_parameter_count()):
            list_param.append(self.circuit.get_parameter(i))
        num_param = self.circuit.get_parameter_count()
        # Initialize a numpy array to record the QFIM
        qfim = np.zeros((num_param, num_param))
        # Assign the signs corresponding to the four terms in a QFIM element
        list_sign = [[1 / 2, 1 / 2], [1 / 2, -1 / 2], [-1 / 2, 1 / 2], [-1 / 2, -1 / 2]]
        # Run the circuit and record the current state vector
        psi = QuantumState(self.circuit.get_qubit_count())
        self.circuit.update_quantum_state(psi)
        # For each QFIM element
        for i in range(0, num_param):
            for j in range(i, num_param):
                # For each term in each element
                for sign_i, sign_j in list_sign:
                    # Shift the parameters by pi/2 * sign
                    list_param[i] += np.pi / 2 * sign_i
                    list_param[j] += np.pi / 2 * sign_j
                    # Update the parameters in the circuit
                    self.update_param(list_param)
                    # Run the shifted circuit and record the shifted state vector
                    psi_shift = QuantumState(self.circuit.get_qubit_count())
                    self.circuit.update_quantum_state(psi_shift)
                    # Calculate each term as the fidelity with a sign factor
                    qfim[i][j] += (
                        abs(inner_product(psi_shift, psi)) ** 2 * sign_i * sign_j * -1
                    )
                    # De-shift the parameters
                    list_param[i] -= np.pi / 2 * sign_i
                    list_param[j] -= np.pi / 2 * sign_j
                    self.update_param(list_param)
                if i != j:
                    # The QFIM is symmetric
                    qfim[j][i] = qfim[i][j]

        return qfim
