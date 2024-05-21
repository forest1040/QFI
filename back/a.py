import math
import numpy as np
from qulacs import (
    QuantumCircuit,
    ParametricQuantumCircuit,
    QuantumState,
    gate,
)
from qulacs.state import inner_product

# initialization
counts_X = 50
X_list = [4 * np.pi * np.random.random((2,8)) for i in range(counts_X)]



