from fisher import QuantumFisher
import numpy as np
from qulacs import ParametricQuantumCircuit, QuantumState


def circuit_circuit():
    circuit = ParametricQuantumCircuit(1)
    circuit.add_H_gate(0)
    circuit.add_parametric_RZ_gate(0, 0)
    circuit.add_parametric_RX_gate(0, 0)
    return circuit


circuit = circuit_circuit()
# print(cir)
qf = QuantumFisher(circuit)

param_list = [[np.pi / 4, 0.1], [np.pi, 0.1], [np.pi / 2, 0.1]]
correct_values = [[[1, 0], [0, 0.5]], [[1, 0], [0, 0]], [[1, 0], [0, 1]]]

for i, param in enumerate(param_list):
    circuit.set_parameter(0, -1 * param[0])
    circuit.set_parameter(1, -1 * param[1])
    # Calculate the QFIM
    qfim = qf.get_qfisher_matrix()
    print(f"The QFIM at {np.array(param)} is \n {qfim.round(14)}.")
    np.testing.assert_allclose(qfim, correct_values[i], atol=1e-3)
