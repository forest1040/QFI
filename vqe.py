from quri_parts.core.operator import Operator, pauli_label

op = Operator(
    {
        pauli_label("X0 Y1"): 0.5 + 0.5j,
        pauli_label("Z0 X1"): 0.2,
    }
)

from math import pi
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit, CONST

param_circuit = LinearMappedUnboundParametricQuantumCircuit(2)
param_circuit.add_H_gate(0)
param_circuit.add_CNOT_gate(0, 1)

theta, phi = param_circuit.add_parameters("theta", "phi")
param_circuit.add_ParametricRX_gate(0, {theta: 1 / 2, phi: 1 / 3, CONST: pi / 2})
param_circuit.add_ParametricRZ_gate(1, {theta: 1 / 3, phi: -1 / 2, CONST: -pi / 2})

from quri_parts.core.state import ParametricCircuitQuantumState

param_state = ParametricCircuitQuantumState(2, param_circuit)

from quri_parts.core.estimator.gradient import create_numerical_gradient_estimator
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_parametric_estimator,
)

qulacs_concurrent_parametric_estimator = (
    create_qulacs_vector_concurrent_parametric_estimator()
)
gradient_estimator = create_numerical_gradient_estimator(
    qulacs_concurrent_parametric_estimator,
    delta=1e-4,
)

gradient = gradient_estimator(op, param_state, [0.2, 0.3])
print("Estimated gradient:", gradient.values)

from quri_parts.core.estimator.gradient import create_parameter_shift_gradient_estimator
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_parametric_estimator,
)

qulacs_concurrent_parametric_estimator = (
    create_qulacs_vector_concurrent_parametric_estimator()
)
gradient_estimator = create_parameter_shift_gradient_estimator(
    qulacs_concurrent_parametric_estimator,
)

gradient = gradient_estimator(op, param_state, [0.2, 0.3])
print("Estimated gradient:", gradient.values)

from quri_parts.circuit.parameter_shift import ShiftedParameters
from quri_parts.core.state import ParametricCircuitQuantumState


def get_raw_param_state_and_shifted_parameters(state, params):
    param_mapping = state.parametric_circuit.param_mapping
    raw_circuit = state.parametric_circuit.primitive_circuit()
    parameter_shift = ShiftedParameters(param_mapping)
    derivatives = parameter_shift.get_derivatives()
    shifted_parameters = [
        d.get_shifted_parameters_and_coef(params) for d in derivatives
    ]

    raw_param_state = ParametricCircuitQuantumState(state.qubit_count, raw_circuit)

    return raw_param_state, shifted_parameters


# Example
raw_state, shifted_params_and_coefs = get_raw_param_state_and_shifted_parameters(
    param_state, [0.2, 0.3]
)

for i, params_and_coefs in enumerate(shifted_params_and_coefs):
    print(f"Parameter shifts for circuit parameter {i}:")
    for p, c in params_and_coefs:
        print(f"  gate params: {p}, coefficient: {c}")

from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_parametric_estimator,
)


def get_parameter_shift_gradient(op, raw_state, shifted_params_and_coefs):
    # Collect gate parameters to be evaluated
    gate_params = set()
    for params_and_coefs in shifted_params_and_coefs:
        for p, _ in params_and_coefs:
            gate_params.add(p)
    gate_params_list = list(gate_params)

    # Prepare a parametric estimator
    estimator = create_qulacs_vector_concurrent_parametric_estimator()

    # Estimate the expectation values
    estimates = estimator(op, raw_state, gate_params_list)
    estimates_dict = dict(zip(gate_params_list, estimates))

    # Sum up the expectation values with the coefficients multiplied
    gradient = []
    for params_and_coefs in shifted_params_and_coefs:
        g = 0.0
        for p, c in params_and_coefs:
            g += estimates_dict[p].value * c
        gradient.append(g)

    return gradient


# Example
gradient = get_parameter_shift_gradient(op, raw_state, shifted_params_and_coefs)
print("Estimated gradient:", gradient)

from collections.abc import Sequence
from dataclasses import dataclass


# This is a return type of GradientEstimator
@dataclass
class _Estimates:
    values: Sequence[complex]
    error_matrix = None


def parameter_shift_gradient_estimator(op, state, params):
    raw_state, shifted_params_and_coefs = get_raw_param_state_and_shifted_parameters(
        state, params
    )
    gradient = get_parameter_shift_gradient(op, raw_state, shifted_params_and_coefs)
    return _Estimates(gradient)


# Example
gradient = parameter_shift_gradient_estimator(op, param_state, [0.2, 0.3])
print("Estimated gradient:", gradient.values)


from quri_parts.algo.ansatz import HardwareEfficient

hw_ansatz = HardwareEfficient(qubit_count=4, reps=3)

from quri_parts.core.state import (
    ComputationalBasisState,
    ParametricCircuitQuantumState,
    apply_circuit,
)

cb_state = ComputationalBasisState(4, bits=0b0011)
parametric_state = apply_circuit(hw_ansatz, cb_state)

from quri_parts.algo.optimizer import Adam

# You can pass optional parameters. See the reference for details
adam_optimizer = Adam()

from quri_parts.core.operator import Operator, pauli_label, PAULI_IDENTITY

# This is Jordan-Wigner transformed Hamiltonian of a hydrogen molecule
hamiltonian = Operator(
    {
        PAULI_IDENTITY: 0.03775110394645542,
        pauli_label("Z0"): 0.18601648886230593,
        pauli_label("Z1"): 0.18601648886230593,
        pauli_label("Z2"): -0.2694169314163197,
        pauli_label("Z3"): -0.2694169314163197,
        pauli_label("Z0 Z1"): 0.172976101307451,
        pauli_label("Z0 Z2"): 0.12584136558006326,
        pauli_label("Z0 Z3"): 0.16992097848261506,
        pauli_label("Z1 Z2"): 0.16992097848261506,
        pauli_label("Z1 Z3"): 0.12584136558006326,
        pauli_label("Z2 Z3"): 0.17866777775953396,
        pauli_label("X0 X1 Y2 Y3"): -0.044079612902551774,
        pauli_label("X0 Y1 Y2 X3"): 0.044079612902551774,
        pauli_label("Y0 X1 X2 Y3"): 0.044079612902551774,
        pauli_label("Y0 Y1 X2 X3"): -0.044079612902551774,
    }
)

from quri_parts.qulacs.estimator import create_qulacs_vector_parametric_estimator

estimator = create_qulacs_vector_parametric_estimator()


def cost_fn(param_values):
    estimate = estimator(hamiltonian, parametric_state, param_values)
    return estimate.value.real


import numpy as np
from quri_parts.core.estimator.gradient import create_numerical_gradient_estimator
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_parametric_estimator,
)

qulacs_concurrent_parametric_estimator = (
    create_qulacs_vector_concurrent_parametric_estimator()
)
gradient_estimator = create_numerical_gradient_estimator(
    qulacs_concurrent_parametric_estimator,
    delta=1e-4,
)


def grad_fn(param_values):
    estimate = gradient_estimator(hamiltonian, parametric_state, param_values)
    return np.asarray([g.real for g in estimate.values])


from quri_parts.algo.optimizer import OptimizerStatus


def vqe(init_params, cost_fn, grad_fn, optimizer):
    opt_state = optimizer.get_init_state(init_params)
    while True:
        opt_state = optimizer.step(opt_state, cost_fn, grad_fn)
        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    return opt_state


init_params = [0.1] * hw_ansatz.parameter_count
result = vqe(init_params, cost_fn, grad_fn, adam_optimizer)
print("Optimized value:", result.cost)
print("Optimized parameter:", result.params)
print("Iterations:", result.niter)
print("Cost function calls:", result.funcalls)
print("Gradient function calls:", result.gradcalls)

from scipy.optimize import minimize


def vqe_scipy(init_params, cost_fn, grad_fn, method, options):
    return minimize(cost_fn, init_params, jac=grad_fn, method=method, options=options)


init_params = [0.1] * hw_ansatz.parameter_count
bfgs_options = {
    "gtol": 1e-6,
}
result = vqe_scipy(init_params, cost_fn, grad_fn, "BFGS", bfgs_options)
print(result.message)
print("Optimized value:", result.fun)
print("Optimized parameter:", result.x)
print("Iterations:", result.nit)
print("Cost function calls:", result.nfev)
print("Gradient function calls:", result.njev)
