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

list_param = []
list_param.append(-1 * np.pi / 4)
list_param.append(-1 * 0.1)
for i in range(len(list_param)):
    circuit.set_parameter(i, list_param[i])

# Calculate the QFIM
qfim = qf.get_qfisher_matrix()
print(f'The QFIM at {np.array(list_param)} is \n {qfim.round(14)}.')

correct_value = [[1, 0], [0, 0.5]]
np.testing.assert_allclose(qfim, correct_value, atol=1e-3)
