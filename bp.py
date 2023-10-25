import math
import random
from typing import Callable, List

from qulacs import Observable, ParametricQuantumCircuit, QuantumState, gate
from qulacs.state import inner_product


# bistate: Observableとcirc適用済みの状態
def python_backprop(circ: ParametricQuantumCircuit, obs: Observable) -> List[float]:
    n = circ.get_qubit_count()
    state = QuantumState(n)
    state.set_zero_state()
    circ.update_quantum_state(state)
    bistate = QuantumState(n)
    work = QuantumState(n)
    obs.apply_to_state(work, state, bistate)

    num_gates = circ.get_gate_count()
    inverse_parametric_gate_position = [-1] * num_gates
    for i in range(circ.get_parameter_count()):
        inverse_parametric_gate_position[circ.get_parametric_gate_position(i)] = i
    ans = [0.0] * circ.get_parameter_count()

    # bistates = []
    # astates = []
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
            ans[inverse_parametric_gate_position[i]] = inner_product(
                bistate, astate
            ).real
            # bistates.append(bistate.copy())
            # astates.append(astate.copy())
        agate = gate_now.get_inverse()
        agate.update_quantum_state(bistate)
        agate.update_quantum_state(state)
    return ans  # , bistates, astates
