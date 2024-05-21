from typing import Callable, List, Optional, Tuple
from numpy.typing import NDArray

from functools import reduce

import numpy as np
from numpy.random import Generator, default_rng
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize

from qulacs import (
    Observable,
    ParametricQuantumCircuit,
    QuantumState,
)
from qulacs.gate import DenseMatrix

from backprop import python_backprop


n_outputs = 1
ansatz = None
observables_str = []
observables = []


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


CostFunc = Callable[[List[float], NDArray[np.float_], NDArray[np.float_]], float]
Jacobian = Callable[
    [List[float], NDArray[np.float_], NDArray[np.float_]], NDArray[np.float_]
]


# def run(
#     theta: List[float],
#     x: NDArray[np.float_],
#     y: NDArray[np.float_],
#     maxiter: Optional[int],
# ) -> Tuple[float, List[float]]:
#     result = minimize(
#         cost_func,
#         theta,
#         args=(x, y),
#         method="BFGS",
#         jac=_cost_func_grad,
#         options={"maxiter": maxiter},
#     )
#     loss = result.fun
#     theta_opt = result.x
#     return loss, theta_opt


def run(
    theta: List[float],
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    maxiter: Optional[int],
) -> Tuple[float, List[float]]:
    n_iter_no_change: Optional[int] = 5
    # callback: Optional[Callable[[List[float]], None]] = None
    tolerance: float = 1e-4

    # pr_A = 0.02
    # pr_Bi = 0.8
    # pr_Bt = 0.995
    # pr_ips = 1e-6
    # # Above is hyper parameters.
    # Bix = 0.0
    # Btx = 0.0

    # moment = np.zeros(len(theta))
    # vel = 0
    theta_now = theta
    maxiter *= len(x)
    prev_cost = cost_func(theta_now, x, y)

    no_change = 0
    for iter in range(0, maxiter, 5):
        grad = _cost_func_grad(
            theta_now,
            x[iter % len(x) : iter % len(x) + 5],
            y[iter % len(y) : iter % len(y) + 5],
        )
        # moment = moment * pr_Bi + (1 - pr_Bi) * grad
        # vel = vel * pr_Bt + (1 - pr_Bt) * np.dot(grad, grad)
        # Bix = Bix * pr_Bi + (1 - pr_Bi)
        # Btx = Btx * pr_Bt + (1 - pr_Bt)
        # theta_now -= pr_A / (((vel / Btx) ** 0.5) + pr_ips) * (moment / Bix)
        theta_now -= grad
        if (n_iter_no_change is not None) and (iter % len(x) < 5):
            # if callback is not None:
            #     callback(theta_now)
            now_cost = cost_func(theta_now, x, y)
            print(f"iter: {iter} cost: {now_cost}")
            if prev_cost - tolerance < now_cost:
                no_change = no_change + 1
                if no_change >= n_iter_no_change:
                    break
            else:
                no_change = 0
            prev_cost = now_cost

    loss = cost_func(theta_now, x, y)
    theta_opt = theta_now
    return loss, theta_opt


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

        r = [observables[i].get_expectation_value(state) for i in range(n_outputs)]
        res.append(r)
    return np.array(res)


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


def backprop(theta: List[float], x: float, obs: Observable) -> List[float]:
    circuit = ParametricQuantumCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_RY_gate(i, np.arcsin(x) * 2)
        circuit.add_RZ_gate(i, np.arccos(x * x) * 2)

    circuit.merge_circuit(ansatz)

    # ret = circuit.backprop(obs)
    ret = python_backprop(circuit, obs)
    ans = [0.0] * len(theta)
    for i in range(len(theta)):
        ans[i] += ret[i]

    return ans


def _cost_func_grad(
    theta: List[float],
    x_scaled: NDArray[np.float_],
    y_scaled: NDArray[np.float_],
) -> NDArray[np.float_]:
    # ansatz.update_parameters(theta)
    for i in range(len(theta)):
        ansatz.set_parameter(i, theta[i])

    pred = _predict_inner(x_scaled)
    mto = pred.copy()

    grad = np.zeros(len(theta))

    n_qubit = ansatz.get_qubit_count()
    for h in range(len(x_scaled)):
        backobs = Observable(n_qubit)
        backobs.add_operator(
            2 * (-y_scaled[h] + mto[h][0]) / n_outputs,
            observables_str[0],
        )
        grad = grad + backprop(theta, x_scaled[h], backobs)

    grad /= len(x_scaled)
    return grad


def fit(
    x_train: NDArray[np.float_],
    y_train: NDArray[np.float_],
    maxiter_or_lr: Optional[int] = None,
) -> Tuple[float, List[float]]:
    if observables_str == []:
        for i in range(n_outputs):
            n_qubit = ansatz.get_qubit_count()
            observable = Observable(n_qubit)
            observable.add_operator(1.0, f"Z {i}")
            observables.append(observable)
            ob = "Z " + str(i)
            observables_str.append(ob)

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


def predict(x_test: NDArray[np.float_]) -> NDArray[np.float_]:
    y_pred = _predict_inner(x_test)
    return y_pred


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
depth = 6

# n_qubit = 3
# depth = 6

# n_qubit = 6
# depth = 10
maxiter = 30
ansatz = create_farhi_neven_ansatz(n_qubit, depth)
opt_loss, opt_params = fit(x_train, y_train, maxiter)
print("trained parameters", opt_params)
print("loss", opt_loss)


y_pred = predict(x_test)
# print(y_pred)
# print(y_pred[:5])

plt.plot(x_test, y_test, "o", label="Test")
plt.plot(
    np.sort(np.array(x_test).flatten()),
    np.array(y_pred)[np.argsort(np.array(x_test).flatten())],
    label="Prediction",
)
plt.legend()
# plt.show()
plt.savefig("qclr.png")
