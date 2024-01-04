import numpy as np
from scipy.optimize import minimize

from quri_parts.core.state import (
    ComputationalBasisState,
    ParametricCircuitQuantumState,
    apply_circuit,
)
from quri_parts.core.operator import Operator, pauli_label, PAULI_IDENTITY
from quri_parts.core.estimator.gradient import (
    create_numerical_gradient_estimator,
    create_parameter_shift_gradient_estimator,
)
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_parametric_estimator,
    create_qulacs_vector_parametric_estimator,
)
from quri_parts.algo.ansatz import HardwareEfficient

hw_ansatz = HardwareEfficient(qubit_count=4, reps=3)
cb_state = ComputationalBasisState(4, bits=0b0011)
parametric_state = apply_circuit(hw_ansatz, cb_state)

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
# estimator = create_qulacs_vector_concurrent_parametric_estimator()
estimator = create_qulacs_vector_parametric_estimator()

qulacs_concurrent_parametric_estimator = (
    create_qulacs_vector_concurrent_parametric_estimator()
)
# gradient_estimator = create_numerical_gradient_estimator(
#     qulacs_concurrent_parametric_estimator,
#     delta=1e-4,
# )
gradient_estimator = create_parameter_shift_gradient_estimator(
    qulacs_concurrent_parametric_estimator,
)


def cost_fn(param_values):
    estimate = estimator(hamiltonian, parametric_state, param_values)
    return estimate.value.real


def grad_fn(param_values):
    estimate = gradient_estimator(hamiltonian, parametric_state, param_values)
    return np.asarray([g.real for g in estimate.values])


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
