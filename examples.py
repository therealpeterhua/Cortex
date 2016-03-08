# This file contains examples from the README. Execute it directly via `python examples.py

from cortex import NeuralNet, LogReg, LnrReg



#------ Neural net: XOR function ---------

data = [
    {'input': [1, 0], 'output': [1]},     # also accepts [1, 0, 1]
    {'input': [0, 1], 'output': [1]},
    {'input': [0, 0], 'output': [0]},
    {'input': [1, 1], 'output': [0]},
]

options = {
    'hidden_sizes': [3, 2],
    'log_progress': True
}

net = NeuralNet()       # also accepts `net = NeuralNet(data)`
net.load_data(data)
net.train(options)

print net.run([0, 1])     # 0.979
print net.run([1, 1])     # 0.029



#------ Linear regression:  y = 2 + 4(x1) + 3(x2) function ---------

data = [
  {'input': [1, 1], 'output': [9]},      # also accepts [1, 1, 9]
  {'input': [2, 3], 'output': [19]},
  {'input': [-5, 2], 'output': [-12]},
  {'input': [3, -4], 'output': [2]}
]

options = {'log_progress': True}

line_regression = LnrReg()         # also accepts `net = LnrReg(data)`
line_regression.load_data(data)
line_regression.train(options)

print line_regression.run([2, 2])        # 15.999



#------ Logistic regression:  x1 = x2 decision boundary ---------

data = [
    [1, 0.9, 0],      # x1 = 1, x2 = 0.9, y = 0
    [5, 4, 0],        # also accepts {'input': [1, 0.9], 'output': [0]}
    [6, 1, 0],
    [8, 7, 0],
    [1, 3, 1],
    [1.1, 1.3, 1],
    [5, 6, 1],
    [6, 6.1, 1],
    [4, 4.5, 1]
]

log_regression = LogReg(data)
log_regression.train()

print log_regression.run([1, 0.5])      # 0.01
print log_regression.run([5, 4])        # 0.00
print log_regression.run([9, 10])       # 1.00
print log_regression.run([10, 15])      # 1.00
