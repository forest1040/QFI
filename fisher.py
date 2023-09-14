# The following were used as references.
# https://github.com/PaddlePaddle/Quantum/blob/master/paddle_quantum/fisher.py

from typing import Optional, Tuple, Union, List
import numpy as np

from tqdm import tqdm
from scipy.special import logsumexp

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
        list_sign = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        # list_sign = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
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
                    # self.cir.update_param(list_param)
                    # Run the shifted circuit and record the shifted state vector
                    # psi_shift = self.cir().numpy()
                    psi_shift = QuantumState(self.circuit.get_qubit_count())
                    self.circuit.update_quantum_state(psi_shift)
                    # Calculate each term as the fidelity with a sign factor
                    qfim[i][j] += (
                        abs(inner_product(psi_shift, psi)) ** 2
                        * sign_i
                        * sign_j
                        * (-0.5)
                        * 0.25
                    )
                    # De-shift the parameters
                    list_param[i] -= np.pi / 2 * sign_i
                    list_param[j] -= np.pi / 2 * sign_j
                    self.update_param(list_param)
                if i != j:
                    # The QFIM is symmetric
                    qfim[j][i] = qfim[i][j]

        return qfim

    # def get_qfisher_norm(self, direction: np.ndarray, step_size: Optional[float] = 0.01) -> float:
    #     # Get the real-time parameters
    #     list_param = self.cir.param
    #     # Run the circuit and record the current state vector
    #     psi = self.cir().numpy()
    #     # Check whether the length of the input direction vector is equal to the number of the variational parameters
    #     assert len(list_param) == len(
    #         direction
    #     ), "the length of direction vector should be equal to the number of the parameters"
    #     # Shift the parameters by step_size * direction
    #     array_params_shift = np.array(
    #         list_param) + np.array(direction) * step_size
    #     # Update the parameters in the circuit
    #     self.cir.update_param(array_params_shift)
    #     # Run the shifted circuit and record the shifted state vector
    #     psi_shift = self.cir().numpy()
    #     # Calculate quantum Fisher-Rao norm along the given direction
    #     qfisher_norm = (1 - abs(np.vdot(psi_shift, psi))**2) * 4 / step_size**2
    #     # De-shift the parameters and update
    #     self.cir.update_param(list_param)

    #     return qfisher_norm

    # def get_eff_qdim(self, num_param_samples: Optional[int] = 4, tol: Optional[float] = None) -> int:
    #     # Get the real-time parameters
    #     list_param = self.cir.param.tolist()
    #     num_param = len(list_param)
    #     # Generate random parameters
    #     param_samples = 2 * np.pi * np.random.random(
    #         (num_param_samples, num_param))
    #     # Record the ranks
    #     list_ranks = []
    #     # Here it has been assumed that the set of points that do not maximize the rank of QFIMs, as singularities, form a null set.
    #     # Thus one can find the maximal rank using a few samples.
    #     for param in param_samples:
    #         # Set the random parameters
    #         self.cir.update_param(param)
    #         # Calculate the ranks
    #         list_ranks.append(self.get_qfisher_rank(tol))
    #     # Recover the original parameters
    #     self.cir.update_param(list_param)

    #     return max(list_ranks)

    # def get_qfisher_rank(self, tol: Optional[float] = None) -> int:
    #     qfisher_rank = np.linalg.matrix_rank(self.get_qfisher_matrix().astype('float64'),
    #                                          1e-6,
    #                                          hermitian=True)
    #     return qfisher_rank
