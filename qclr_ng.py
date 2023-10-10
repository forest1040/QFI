from typing import Callable, List, Optional, Tuple
from numpy.typing import NDArray

import copy
from pprint import pprint
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

from fisher import QuantumFisher

# from back_fisher import fisher

from slove import lu_decomposition, multiply_permutation_matrix, backward_substitution

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


def calc_fisher(theta, x_scaled):
    result = np.zeros((len(theta), len(theta)), dtype=float)
    for x in x_scaled:
        circuit = ParametricQuantumCircuit(n_qubit)
        for i in range(n_qubit):
            circuit.add_RY_gate(i, np.arcsin(x) * 2)
            circuit.add_RZ_gate(i, np.arccos(x * x) * 2)

        for i in range(len(theta)):
            ansatz.set_parameter(i, theta[i])

        circuit.merge_circuit(ansatz)
        qf = QuantumFisher(circuit)
        result += qf.get_qfisher_matrix()
        # result = fisher(circuit, observables[0])
    return result / len(x_scaled)


# 目的行の列が0だった場合に入れ替える
def swap_row(mtx, rev_mtx, row, col):
    num = mtx[row][col]
    if num == 0.0:
        org_vec = mtx[row]
        org_rev_vec = rev_mtx[row]
        for trg_row in range(row + 1, len(mtx)):
            trg_vec = mtx[trg_row]
            if trg_vec[col] == 0.0:
                continue
            temp_vec = mtx[trg_row]
            temp_rev_vec = rev_mtx[trg_row]
            mtx[row] = temp_vec
            mtx[trg_row] = org_vec
            rev_mtx[row] = temp_rev_vec
            rev_mtx[trg_row] = org_rev_vec
            break
        # 入れ替え先が全部0.0だったらNoneを返す
        else:
            return None, None
    return mtx, rev_mtx


# 掃き出し法
def calc_inv_matrix(mtx):
    row_count = len(mtx)
    org_mtx = copy.copy(mtx)
    # 単位行列で逆行列を初期化
    rev_mtx = []
    for i in range(row_count):
        rev_vec = [0.0] * row_count
        rev_vec[i] = 1.0
        rev_mtx.append(rev_vec)
    # 掃き出し開始
    for i in range(row_count):  # 行ループ
        # 行入れ替え
        mtx, rev_mtx = swap_row(mtx, rev_mtx, i, i)
        # 入れ替え先も全部0.0だったらエラー
        if mtx is None:
            print("error :")
            return None
        # 現在の基準行をそれぞれ取得
        base_vec = mtx[i]
        base_rev_vec = rev_mtx[i]
        # 目的行の目的列が1.0になるように逆数を掛ける
        if base_vec[i] != 1.0:
            inv_num = 1.0 / base_vec[i]  # 逆数
            base_vec = list(map(lambda x: x * inv_num, base_vec))
            mtx[i] = base_vec
            base_rev_vec = list(map(lambda x: x * inv_num, rev_mtx[i]))
            rev_mtx[i] = base_rev_vec
        # 他の行の列を0にする
        for j in range(row_count):
            if i == j:  # 現在の基準行はスキップ
                continue
            trg_vec = mtx[j]
            trg_rev_vec = rev_mtx[j]
            # すでに0になっていたらスキップ
            if trg_vec[i] == 0:
                continue
            # 0にするための比率を算出
            base_num = base_vec[i]
            trg_num = trg_vec[i]
            ratio = trg_num / base_num
            new_rev_vec = []
            new_vec = []
            # 他の列に対しては同様の比率で引き算する
            for k in range(row_count):
                # 元の行列を定数倍して引き算
                base_num = base_vec[k]
                trg_num = trg_vec[k]
                new_num = trg_num - (base_num * ratio)
                new_vec.append(new_num)
                # 逆行列も同様に定数倍して引き算
                base_rev_num = base_rev_vec[k]
                trg_rev_num = trg_rev_vec[k]
                new_rev_num = trg_rev_num - (base_rev_num * ratio)
                new_rev_vec.append(new_rev_num)
            # 補正後の行ベクトルに置き換え
            rev_mtx[j] = new_rev_vec
            mtx[j] = new_vec
    return rev_mtx


def run(
    theta: List[float],
    x: NDArray[np.float_],
    y: NDArray[np.float_],
    maxiter: Optional[int],
) -> Tuple[float, List[float]]:
    n_iter_no_change: Optional[int] = 5
    tolerance: float = 1e-4

    eta = 0.9

    theta_now = theta
    maxiter *= len(x)
    prev_cost = cost_func(theta_now, x, y)

    no_change = 0

    batch_size = 1

    for iter in range(0, maxiter, batch_size):
        grad = _cost_func_grad(
            theta_now,
            x[iter % len(x) : iter % len(x) + batch_size],
            y[iter % len(y) : iter % len(y) + batch_size],
        )
        F = calc_fisher(theta_now, x[iter % len(x) : iter % len(x) + batch_size])
        # theta_now -= eta * np.linalg.inv(F) @ grad
        # theta_now -= F @ grad
        # pprint(calc_inv_matrix(F))
        # theta_now -= grad
        # theta_now -= np.dot(np.linalg.inv(F), grad)
        # theta_now -= np.linalg.solve(F, grad)
        # theta_now -= np.linalg.solve(F, np.eye(len(grad))) @ grad

        N = len(theta_now)
        L, U, P = lu_decomposition(F, N)
        # LY = PBを解く
        L = [row[::-1] for row in L[:]][::-1]  # 後退代入と上下逆なので逆順に
        PB = multiply_permutation_matrix(P, grad, N)[::-1]  # 置換行列をかけて後退代入と上下逆なので逆順に
        Y = backward_substitution(L, PB, N)[::-1]
        # UX = Yを解く
        X = backward_substitution(U, Y, N)
        theta_now -= np.array(X)

        if (n_iter_no_change is not None) and (iter % len(x) < batch_size):
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
plt.savefig("qclr_ng.png")
