from qulacs import ParametricQuantumCircuit, GradCalculator, Observable

from backprop import python_backprop

n = 2
observable = Observable(n)
observable.add_operator(1.0, "X 0")
circuit = ParametricQuantumCircuit(n)

theta = [2.2, 1.4, 0.8]
circuit.add_parametric_RX_gate(0, theta[0])
circuit.add_parametric_RY_gate(0, theta[1])
circuit.add_parametric_RZ_gate(0, theta[2])

# GradCalculatorの場合
gcalc = GradCalculator()
print(gcalc.calculate_grad(circuit, observable))
# 第三引数に回転角を指定することも可能
print(gcalc.calculate_grad(circuit, observable, theta))
# Backpropを使って求めた場合
print(circuit.backprop(observable))
print(python_backprop(circuit, observable))
