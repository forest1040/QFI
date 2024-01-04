from typing import List, Optional, Tuple
from numpy.typing import NDArray

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

from qulacs import (
    Observable,
    ParametricQuantumCircuit,
    QuantumState,
)


ansatz = None


def create_farhi_neven_ansatz(
    n_qubit: int, c_depth: int, seed: Optional[int] = 0
) -> ParametricQuantumCircuit:
    circuit = ParametricQuantumCircuit(n_qubit)
    zyu = list(range(n_qubit))
    rng = default_rng(seed)
    for _ in range(c_depth):
        rng.shuffle(zyu)

        for i in range(0, n_qubit - 1, 2):
            angle_x = 2.0 * np.pi * rng.random()
            angle_y = 2.0 * np.pi * rng.random()
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_parametric_RX_gate(zyu[i], angle_x)
            circuit.add_parametric_RY_gate(zyu[i], angle_y)
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            angle_x = 2.0 * np.pi * rng.random()
            angle_y = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RY_gate(zyu[i], -angle_y)
            circuit.add_parametric_RX_gate(zyu[i], -angle_x)
    return circuit


def _predict_inner(x_scaled: NDArray[np.float_]) -> NDArray[np.float_]:
    res = []
    for x in x_scaled:
        n_qubit = ansatz.get_qubit_count()
        state = QuantumState(n_qubit)
        state.set_zero_state()

        circuit = ParametricQuantumCircuit(n_qubit)
        for i in range(n_qubit):
            circuit.add_RY_gate(i, np.arcsin(x) * 2)
            circuit.add_RZ_gate(i, np.arccos(x * x) * 2)

        circuit.merge_circuit(ansatz)
        circuit.update_quantum_state(state)

        observable = Observable(n_qubit)
        observable.add_operator(1.0, "Z 0")
        r = [observable.get_expectation_value(state)]
        res.append(r)
    return np.array(res)


def predict(x_test: NDArray[np.float_]) -> NDArray[np.float_]:
    y_pred = _predict_inner(x_test)
    return y_pred


def cost_func(
    theta: List[float],
    x_scaled: NDArray[np.float_],
    y_scaled: NDArray[np.float_],
) -> float:
    for i in range(len(theta)):
        ansatz.set_parameter(i, theta[i])

    y_pred = _predict_inner(x_scaled)
    cost = mean_squared_error(y_pred, y_scaled)
    return cost


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
    )
    loss = result.fun
    theta_opt = result.x
    return loss, theta_opt


def fit(
    x_train: NDArray[np.float_],
    y_train: NDArray[np.float_],
    maxiter_or_lr: Optional[int] = None,
) -> Tuple[float, List[float]]:
    theta_init = []
    for i in range(ansatz.get_parameter_count()):
        param = ansatz.get_parameter(i)
        theta_init.append(param)

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
    return np.array(x_train), np.array(y_train)


x_min = -1.0
x_max = 1.0
num_x = 80
x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)
x_test, y_test = generate_noisy_sine(x_min, x_max, num_x)

n_qubit = 4
depth = 2

# n_qubit = 3
# depth = 6

# n_qubit = 6
# depth = 10

# maxiter = 2000
maxiter = 1000
ansatz = create_farhi_neven_ansatz(n_qubit, depth)
opt_loss, opt_params = fit(x_train, y_train, maxiter)
print("trained parameters", opt_params)
print("loss", opt_loss)

y_pred = predict(x_test)

plt.plot(x_test, y_test, "o", label="Test")
plt.plot(
    np.sort(np.array(x_test).flatten()),
    np.array(y_pred)[np.argsort(np.array(x_test).flatten())],
    label="Prediction",
)
plt.legend()
# plt.show()
plt.savefig("qclr.png")
