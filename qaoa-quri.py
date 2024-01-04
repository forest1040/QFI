# https://dojo.qulacs.org/ja/latest/notebooks/5.3_quantum_approximate_optimazation_algorithm.html
# from qulacs import QuantumState, QuantumCircuit, Observable, PauliOperator
# from qulacs.gate import H, CNOT, RX, RZ
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState

from quri_parts.qulacs.estimator import create_qulacs_vector_estimator
from quri_parts.qulacs.sampler import create_qulacs_vector_sampler

from scipy.optimize import minimize
import numpy as np


estimator = create_qulacs_vector_estimator()
sampler = create_qulacs_vector_sampler()

## 頂点の数
n = 4

## C(Z)をqulacs.Observableとして定義
# cost_observable = Observable(n)
# for i in range(n):
#     cost_observable.add_operator(
#         PauliOperator("Z {:} Z {:}".format(i, (i + 1) % n), 0.5)
#     )
cost_observable = Operator(
    {
        pauli_label("Z0 Z1"): 0.5,
        pauli_label("Z1 Z2"): 0.5,
        pauli_label("Z2 Z3"): 0.5,
        pauli_label("Z3 Z0"): 0.5,
    }
)


# circuit に U_C(gamma) を加える関数
def add_U_C(circuit, gamma):
    for i in range(n):
        j = (i + 1) % n
        circuit.add_CNOT_gate(i, j)
        # circuit.add_gate(RZ(j, -2 * gamma))  ## qulacsでは RZ(theta)=e^{i*theta/2*Z}
        circuit.add_RZ_gate(j, 2 * gamma)
        circuit.add_CNOT_gate(i, j)
    return circuit


# circuit に U_X(beta) を加える関数
def add_U_X(circuit, beta):
    for i in range(n):
        # circuit.add_gate(RX(i, -2 * beta))
        circuit.add_RX_gate(i, 2 * beta)
    return circuit


# p=1 の |beta, gamma> を作って <beta, gamma| C(Z) |beta, gamma> を返す関数
# x = [beta, gamma]
def QAOA_output_onelayer(x):
    beta, gamma = x

    circuit = QuantumCircuit(n)
    ## 重ね合わせを作るため、アダマールゲートをかける
    for i in range(n):
        circuit.add_H_gate(i)
    ## U_C, U_Xをかける
    circuit = add_U_C(circuit, gamma)
    circuit = add_U_X(circuit, beta)

    ## |beta, gamma>を作る
    # state = GeneralCircuitQuantumState(n)
    # state.set_zero_state()
    # state = ComputationalBasisState(n, bits=0b0000)
    circuit_state = GeneralCircuitQuantumState(n, circuit)
    # circuit.update_quantum_state(state)
    # return cost_observable.get_expectation_value(state)
    return estimator(cost_observable, circuit_state).value.real


## 初期値
x0 = np.array([0.1, 0.1])

## scipy.minimize を用いて最小化
result = minimize(QAOA_output_onelayer, x0, options={"maxiter": 500}, method="powell")
print(result.fun)  # 最適化後の値
print(result.x)  # 最適化後の(beta, gamma)


# 最適なbeta, gammaを使って |beta, gamma> をつくる
beta_opt, gamma_opt = result.x

circuit = QuantumCircuit(n)
## 重ね合わせを作るため、アダマールゲートをかける
for i in range(n):
    circuit.add_H_gate(i)
## U_C, U_Xをかける
circuit = add_U_C(circuit, gamma_opt)
circuit = add_U_X(circuit, beta_opt)

## |beta, gamma>を作る
# state = GeneralCircuitQuantumState(n)
# state.set_zero_state()
# state = ComputationalBasisState(n, bits=0b0000)
# circuit.update_quantum_state(state)
## z方向に観測した時の確率分布を求める. (状態ベクトルの各成分の絶対値の二乗=観測確率)
# probs = np.abs(state.get_vector()) ** 2
# print(probs)

# circuit_state = GeneralCircuitQuantumState(n, circuit)
sampling_result = sampler(circuit, shots=1000)

z_basis = []
counts = []
for bits, count in sampling_result.items():
    z_basis.append(bits)
    counts.append(count / 1000 * 100)  # shots / %

zipped = zip(z_basis, counts)
data = sorted(zipped, key=lambda x: x[0])

z_basis = []
counts = []
for bits, count in data:
    z_basis.append(format(bits, "04b"))
    counts.append(count)

# probs = [float(x / 1000) for x in sampling_result.values()]
# print(probs)
# ## z方向に射影測定した時に得られる可能性があるビット列
# z_basis = [format(i, "b").zfill(n) for i in range(len(probs))]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.xlabel("states")
plt.ylabel("probability(%)")
plt.bar(z_basis, counts)
# plt.show()
plt.savefig("qaoa-quri.png")
