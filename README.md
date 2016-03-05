CONGRATS YOU'VE FOUND ME!

This library implements 3 popular prediction / classification techniques.
Specifically, it implements a neural network, and logistic and linear regression, using no external libraries.

Performs batch gradient descent across all techniques, and employs sigmoid activation for the NN and logistic reg.

Bold driver heuristic adjusts learning rate on the fly through a momentum factor (achieving multiples of efficiency gains in some cases), but feel free to turn off if you wish.

Regularization factor addresses overfit by squashing higher-order features with great prejudice. Setting this value high exhibits the same (though disproportionally smaller) "compression" effect on lower-order weights, tending the system towards 0.

Vectorized implementations have been implemented in Octave, and will be ported over to NumPy eventually. In the meantime, please enjoy some for loops and list comprehensions.


Training the XOR function...

```python
data = [
    {'input': [1, 0], 'output': [1]},
    {'input': [0, 1], 'output': [1]},
    {'input': [0, 0], 'output': [0]},
    {'input': [1, 1], 'output': [1]},
]

options = {'momentum': 1.1, 'log': True}

net = NeuralNet()
net.load_data(data)
net.train(options)

net.run([0, 1])     # 0.979
```

<sup>NOTE: Occasionally, your neural nets may return *less accurate* results than anticipated. If so, try training the network again. Batch descent is sensitive to initial conditions and can hang on local minima, but each training call will randomize the initialization weights.</sup>

The lower the learning rate and higher the desired degree of accuracy, the longer the training time, and the greater the chance of overfit. Network training time is a function of the learning rate and desired degree of accuracy.

ELABORATE MORE. Learning rate is dynamic? Or should recommend lowering the degree of accuracy?


TODOs:
- Pruning algo for neural network to "trim" redundant nodes
- Serialization of weights, allowing user to save and resume work on large data sets
- Implement neural network momentum scaling
- Regularization in logistic and linear regression
- More extensive error handling -- empty layers for NN, edge-case inputs
