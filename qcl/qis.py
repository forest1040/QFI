# https://qiita.com/notori48/items/ebfa4a8c8ee2da134ba1

from qiskit.primitives import Estimator
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

import matplotlib.pyplot as plt
import numpy as np


nqubit = 3
depth = 1

# nqubit = 4
# depth = 2

x_min = -1.0
x_max = 1.0
num_x_train = 50

func_to_learn = lambda x: np.sin(x * np.pi)

random_seed = 0
np.random.seed(random_seed)

# 教師データを準備
x_train = x_min + (x_max - x_min) * np.random.rand(num_x_train)
y_train = func_to_learn(x_train)

# 現実のデータを用いる場合を想定し、きれいなsin関数にノイズを付加
mag_noise = 0.05
y_train = y_train + mag_noise * np.random.randn(num_x_train)

obs_op = SparsePauliOp(["ZII"], [1.0])


def add_xin(circuit_in, x_in):
    angle_y = np.arcsin(x_in)
    angle_z = np.arccos(x_in**2)
    for i in range(nqubit):
        circuit_in.ry(angle_y, i)
        circuit_in.rz(angle_z, i)
    circuit_in.barrier()
    return circuit_in


# def add_Variational_Circuit(circuit_in, i):
#     for j in range(nqubit):
#         v = Parameter("theta[%s]" % str(3 * j + 3 * nqubit * i + 0))
#         circuit_in.rx(v, j)
#         v = Parameter("theta[%s]" % str(3 * j + 3 * nqubit * i + 1))
#         circuit_in.rz(v, j)
#         v = Parameter("theta[%s]" % str(3 * j + 3 * nqubit * i + 2))
#         circuit_in.rx(v, j)
#     circuit_in.barrier()
#     return circuit_in


def Make_Circuit_1(circuit_in, x_in):
    circuit_in = add_xin(circuit_in, x_in)
    circuit_in.barrier()
    return circuit_in


def Make_Circuit_2(circuit_in):
    idx = 0
    for _ in range(depth):
        circuit_in.barrier()
        # circuit_in = add_Variational_Circuit(circuit_in, i)
        for i in range(nqubit):
            for j in range(i + 1, nqubit):
                # if i == j:
                #     continue
                v = Parameter(f"theta[{idx}]")
                idx += 1
                circuit_in.rx(v, i)
                circuit_in.cx(i, j)
                v = Parameter(f"theta[{idx}]")
                idx += 1
                circuit_in.rz(v, i)
                v = Parameter(f"theta[{idx}]")
                idx += 1
                circuit_in.cx(i, j)
                circuit_in.rx(v, i)
        circuit_in.barrier()
    return circuit_in


param_size = 3 * nqubit * depth
theta_init = [2.0 * np.pi * np.random.rand() for i in range(param_size)]


def qcl_pred(x_in, theta_in):
    circuit = QuantumCircuit(nqubit)
    circuit = Make_Circuit_1(circuit, x_in)
    circuit = Make_Circuit_2(circuit)
    estimator = Estimator()
    job = estimator.run(
        circuits=[circuit], observables=[obs_op], parameter_values=[theta_in]
    )
    result = job.result()
    exp_val = result.values
    # print('exp_val: ', exp_val)
    return exp_val


def cost(theta_in):
    y_pred = [qcl_pred(x, theta_in) for x in x_train]
    # quadratic loss
    L = ((y_pred - y_train) ** 2).mean()
    return L


iter = 0


def callback(xk):
    global iter
    print("callback {}: xk={}".format(iter, xk))
    iter += 1
    # print("callback : xk={}".format(xk))


from scipy.optimize import minimize

# 学習 (筆者のPCで数十分程度かかる)
result = minimize(
    cost, theta_init, method="Nelder-Mead", callback=callback, options={"maxiter": 100}
)
# result = minimize(cost, theta_init, method='Powell')

# 最適化後のcost_functionの値
result.fun
theta_opt = result.x

# プロット
plt.figure(figsize=(10, 6))

xlist = np.arange(x_min, x_max, 0.02)

# 教師データ
plt.plot(x_train, y_train, "o", label="Teacher")

# モデルの予測値
y_pred = np.array([qcl_pred(x, theta_opt) for x in xlist])
plt.plot(xlist, y_pred, label="Final Model Prediction")

plt.legend()
# plt.show()
plt.savefig("qis.png")
