import math
import random
from typing import Callable, List

from qulacs import Observable, ParametricQuantumCircuit, QuantumState, gate
from qulacs.state import inner_product


@profile
def cpp_backprop_inner_product(
    circ: ParametricQuantumCircuit, bistate: QuantumState
) -> List[float]:
    return circ.backprop_inner_product(bistate)


@profile
def python_backprop_inner_product(
    circ: ParametricQuantumCircuit, bistate: QuantumState
) -> List[float]:
    n = circ.get_qubit_count()
    state = QuantumState(n)
    state.set_zero_state()
    circ.update_quantum_state(state)

    num_gates = circ.get_gate_count()
    inverse_parametric_gate_position = [-1] * num_gates
    for i in range(circ.get_parameter_count()):
        inverse_parametric_gate_position[circ.get_parametric_gate_position(i)] = i
    ans = [0.0] * circ.get_parameter_count()

    astate = QuantumState(n)
    for i in range(num_gates - 1, -1, -1):
        gate_now = circ.get_gate(i)
        if inverse_parametric_gate_position[i] != -1:
            astate.load(state)
            if gate_now.get_name() == "ParametricRX":
                rcpi = gate.RX(gate_now.get_target_index_list()[0], math.pi)
            elif gate_now.get_name() == "ParametricRY":
                rcpi = gate.RY(gate_now.get_target_index_list()[0], math.pi)
            elif gate_now.get_name() == "ParametricRZ":
                rcpi = gate.RZ(gate_now.get_target_index_list()[0], math.pi)
            else:
                raise RuntimeError()
            rcpi.update_quantum_state(astate)
            ans[inverse_parametric_gate_position[i]] = (
                inner_product(bistate, astate).real / 2.0
            )
        agate = gate_now.get_inverse()
        agate.update_quantum_state(bistate)
        agate.update_quantum_state(state)
    return ans


def backprop(
    circ: ParametricQuantumCircuit,
    obs: Observable,
    backprop_func: Callable[[ParametricQuantumCircuit, QuantumState], List[float]],
) -> List[float]:
    n = circ.get_qubit_count()
    state = QuantumState(n)
    state.set_zero_state()
    circ.update_quantum_state(state)
    bistate = QuantumState(n)
    astate = QuantumState(n)

    obs.apply_to_state(astate, state, bistate)
    bistate.multiply_coef(2)

    ans = backprop_func(circ, bistate)
    return ans


def bench(
    backprop_func: Callable[[ParametricQuantumCircuit, QuantumState], List[float]],
    n: int,
    m: int,
) -> float:
    import time

    circ = ParametricQuantumCircuit(n)
    for _ in range(m):
        if _ % 2 == 0:
            if random.randint(0, 1) == 0:
                circ.add_H_gate(random.randint(0, n - 1))
            else:
                q1 = random.randint(0, n - 1)
                q2 = random.randint(0, n - 2)
                if q2 >= q1:
                    q2 += 1
                circ.add_CNOT_gate(q1, q2)
        else:
            r = random.randint(0, 2)
            t = (random.random() * 2 - 1) * math.pi
            if r == 0:
                circ.add_parametric_RX_gate(random.randint(0, n - 1), t)
            elif r == 1:
                circ.add_parametric_RY_gate(random.randint(0, n - 1), t)
            else:
                circ.add_parametric_RZ_gate(random.randint(0, n - 1), t)

    obs = Observable(n)
    for i in range(n):
        obs.add_operator(random.random(), "XYZ"[random.randint(0, 2)] + " " + str(i))
    st = time.time()
    backprop(circ, obs, backprop_func)
    return time.time() - st


n_qubits = 8
depth = 100000

bench(python_backprop_inner_product, n_qubits, depth)
bench(cpp_backprop_inner_product, n_qubits, depth)
