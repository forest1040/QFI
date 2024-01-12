from typing import List, Optional, Tuple
from numpy.typing import NDArray

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

# from qiskit_ibm_runtime import (
#     QiskitRuntimeService,
#     Sampler,
#     Estimator,
#     Session,
#     Options,
# )
from qiskit.primitives import Estimator

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp


# service = QiskitRuntimeService()
# backend = "ibmq_qasm_simulator"  # use the simulator
# options = Options()
# options.simulator.seed_simulator = 42
# options.execution.shots = 1000
# options.optimization_level = 0  # no optimization
# options.resilience_level = 0  # no error mitigation

# estimator = Estimator(options=options, backend=backend)
estimator = Estimator()

ansatz = None
seed = 0

n_qubit = 4
depth = 2
param_size = 16
PauliOpStr = "ZIII"
# maxiter = 2000
# maxiter = 1000
maxiter = 500
# maxiter = 2

# n_shots = 1000


# def create_farhi_neven_ansatz(
#     n_qubit: int, c_depth: int, seed: Optional[int] = 0
# ) -> QuantumCircuit:
#     circuit = QuantumCircuit(n_qubit)
#     zyu = list(range(n_qubit))
#     rng = default_rng(seed)
#     idx = 0
#     for _ in range(c_depth):
#         rng.shuffle(zyu)
#         for i in range(0, n_qubit - 1, 2):
#             param_rx = Parameter(f"theta[{idx}]")
#             idx += 1
#             param_ry = Parameter(f"theta[{idx}]")
#             idx += 1
#             circuit.cx(zyu[i + 1], zyu[i])
#             circuit.rx(param_rx, zyu[i])
#             circuit.ry(param_ry, zyu[i])
#             circuit.cx(zyu[i + 1], zyu[i])
#             circuit.ry(-1 * param_ry, zyu[i])
#             circuit.rx(-1 * param_rx, zyu[i])
#     return circuit


def create_farhi_neven_ansatz(
    n_qubit: int, c_depth: int, seed: Optional[int] = 0
) -> QuantumCircuit:
    circuit = QuantumCircuit(n_qubit)
    zyu = list(range(n_qubit))
    rng = default_rng(seed)
    idx = 0
    for _ in range(c_depth):
        rng.shuffle(zyu)
        for i in range(0, n_qubit - 1, 2):
            circuit.cx(zyu[i + 1], zyu[i])
            param_rx = Parameter(f"theta[{idx}]")
            idx += 1
            param_ry = Parameter(f"theta[{idx}]")
            idx += 1
            circuit.rx(param_rx, zyu[i])
            circuit.ry(param_ry, zyu[i])
            circuit.cx(zyu[i + 1], zyu[i])
            param_rx = Parameter(f"theta[{idx}]")
            idx += 1
            param_ry = Parameter(f"theta[{idx}]")
            idx += 1
            circuit.ry(param_ry, zyu[i])
            circuit.rx(param_rx, zyu[i])
    return circuit


def _predict_inner(
    x_scaled: NDArray[np.float_], theta: List[float]
) -> NDArray[np.float_]:
    res = []
    for x in x_scaled:
        circuit = QuantumCircuit(n_qubit)

        for i in range(n_qubit):
            circuit.ry(np.arcsin(x) * 2, i)
            circuit.rz(np.arccos(x * x) * 2, i)

        circuit = circuit.compose(ansatz)
        observable = SparsePauliOp([PauliOpStr], [2.0])
        job = estimator.run(
            circuits=[circuit], observables=[observable], parameter_values=[theta]
        )
        result = job.result()
        exp_val = result.values
        res.append(exp_val)

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
    for _ in range(param_size):
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
plt.savefig("qclr-qis.png")
