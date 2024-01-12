from typing import List, Optional, Tuple
from numpy.typing import NDArray

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

from quri_parts.circuit import QuantumCircuit, CNOT, UnboundParametricQuantumCircuit
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState

from quri_parts.qulacs.sampler import create_qulacs_vector_concurrent_sampler
from quri_parts.qulacs.estimator import create_qulacs_vector_concurrent_estimator

from quri_parts.qiskit.backend import QiskitSamplingBackend
from quri_parts.core.sampling import (
    create_sampler_from_sampling_backend,
    create_concurrent_sampler_from_sampling_backend,
)
from quri_parts.core.estimator.sampling import create_sampling_concurrent_estimator
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)

from qiskit_ibm_runtime import QiskitRuntimeService

ansatz = None
n_qubit = 4
depth = 2
seed = 0
n_shots = 1000

# qulacs
if 1:
    sampler = create_qulacs_vector_concurrent_sampler()
    estimator = create_qulacs_vector_concurrent_estimator()
else:
    # IBMQ
    service = QiskitRuntimeService()
    # ibm_algiers / ibm_hanoi /ibmq_kolkata / ibm_cairo
    device = service.backend("ibm_hanoi")
    backend = QiskitSamplingBackend(device)
    # sampler = create_sampler_from_sampling_backend(backend)
    sampler = create_concurrent_sampler_from_sampling_backend(backend)
    allocator = create_equipartition_shots_allocator()
    estimator = create_sampling_concurrent_estimator(
        n_shots, sampler, bitwise_commuting_pauli_measurement, allocator
    )


def create_farhi_neven_ansatz(
    n_qubit: int, c_depth: int, seed: Optional[int] = 0
) -> UnboundParametricQuantumCircuit:
    circuit = UnboundParametricQuantumCircuit(n_qubit)
    zyu = list(range(n_qubit))
    rng = default_rng(seed)
    for _ in range(c_depth):
        rng.shuffle(zyu)
        for i in range(0, n_qubit - 1, 2):
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_ParametricRX_gate(zyu[i])
            circuit.add_ParametricRY_gate(zyu[i])
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_ParametricRY_gate(zyu[i])
            circuit.add_ParametricRX_gate(zyu[i])
    return circuit


def _predict_inner(
    x_scaled: NDArray[np.float_], theta: List[float]
) -> NDArray[np.float_]:
    res = []
    for x in x_scaled:
        circuit = UnboundParametricQuantumCircuit(n_qubit)

        for i in range(n_qubit):
            circuit.add_RY_gate(i, np.arcsin(x) * 2)
            circuit.add_RZ_gate(i, np.arccos(x * x) * 2)

        bind_circuit = ansatz.bind_parameters(theta)
        circuit = circuit.combine(bind_circuit)
        circuit_state = GeneralCircuitQuantumState(n_qubit, circuit)
        observable = Operator(
            {
                pauli_label("Z0"): 2.0,
            }
        )
        v = estimator([observable], [circuit_state])[0].value.real
        # v = estimator(observable, circuit_state).value.real
        res.append(v)

        # sampling_result = sampler(circuit, shots=n_shots)
        # counts = sampling_result
        # # print(counts)
        # # print(counts.get(0))

        # if counts.get(0) is not None:
        #     res.append(float(counts.get(0) / n_shots))
        # else:
        #     res.append(0.0)

    return np.array(res)


def predict(x_test: NDArray[np.float_], theta: List[float]) -> NDArray[np.float_]:
    y_pred = _predict_inner(x_test, theta)
    return y_pred


def cost_func(
    theta: List[float],
    x_scaled: NDArray[np.float_],
    y_scaled: NDArray[np.float_],
) -> float:
    y_pred = _predict_inner(x_scaled, theta)
    cost = mean_squared_error(y_pred, y_scaled)
    return cost


iter = 0


def callback(xk):
    global iter
    # print("callback {}: xk={}".format(iter, xk))


def run(
    theta: List[float],
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    maxiter: Optional[int],
) -> Tuple[float, List[float]]:
    result = minimize(
        cost_func,
        theta,
        args=(x, y),
        method="Nelder-Mead",
        options={"maxiter": maxiter},
        callback=callback,
    )
    loss = result.fun
    theta_opt = result.x
    return loss, theta_opt


def fit(
    x_train: NDArray[np.float_],
    y_train: NDArray[np.float_],
    maxiter_or_lr: Optional[int] = None,
) -> Tuple[float, List[float]]:
    rng = default_rng(seed)
    theta_init = []
    for _ in range(4 * 2 * depth):
        theta_init.append(2.0 * np.pi * rng.random())

    return run(
        theta_init,
        x_train,
        y_train,
        maxiter_or_lr,
    )


def generate_noisy_sine(x_min, x_max, num_x):
    rng = default_rng(0)
    x_train = [[rng.uniform(x_min, x_max)] for _ in range(num_x)]
    y_train = [np.sin(np.pi * x[0]) for x in x_train]
    mag_noise = 0.01
    y_train += mag_noise * rng.random(num_x)
    # return np.array(x_train), np.array(y_train)
    return np.array(x_train).flatten(), np.array(y_train)


x_min = -1.0
x_max = 1.0
num_x = 80
x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)
x_test, y_test = generate_noisy_sine(x_min, x_max, num_x)

ansatz = create_farhi_neven_ansatz(n_qubit, depth, seed)

# maxiter = 2000
# maxiter = 1000
maxiter = 500
# maxiter = 2
opt_loss, opt_params = fit(x_train, y_train, maxiter)
print("trained parameters", opt_params)
print("loss", opt_loss)

y_pred = predict(x_test, opt_params)

plt.plot(x_test, y_test, "o", label="Test")
plt.plot(
    np.sort(np.array(x_test).flatten()),
    np.array(y_pred)[np.argsort(np.array(x_test).flatten())],
    label="Prediction",
)
plt.legend()
# plt.show()
plt.savefig("qclr-quri.png")
