This library is a tool for 3 popular prediction / classification techniques.
It implements a neural network, and logistic and linear regression, using no external libraries.

Performs batch gradient descent across all techniques, and employs sigmoid activation for the NN and logistic reg.
Bold driver heuristic adjusts learning rate on the fly through a momentum factor (achieving multiples of efficiency gains in some cases), but turn off if you wish.
Regularization factor allows you to address overfit by compressing higher-order features.

Vectorized implementations have been implemented in Octave, and will be ported to NumPy eventually. In the meantime, please enjoy some for loops and list comprehensions.

###Neural Network

#####Example: Training the XOR function...
```python
data = [
    {'input': [1, 0], 'output': [1]},
    {'input': [0, 1], 'output': [1]},
    {'input': [0, 0], 'output': [0]},
    {'input': [1, 1], 'output': [1]},
]

options = {'momentum': 1.1, 'log_progress': True}

net = NeuralNet()
net.load_data(data)
net.train(options)

net.run([0, 1])     # 0.979
```

<sup>NOTE: Occasionally, your neural nets may return *higher error* results than anticipated. If so, try training the network again. Batch descent is sensitive to initial conditions and can hang on local minima, but each training call will randomize the starting weights.</sup>

#####Restrictions:
Each element of the output vector must be between 0 and 1. Trains multi-class scenarios via multiple-element output vectors (ie. `[1, 0, 0]`, `[0, 1, 0]`, `[0, 0, 1]` for Class I, Class II, Class III). This multi-class input method will be abstracted away from the API in the future.

######Adjustable model parameters (`options` dict in XOR example)
  - `hidden_sizes`: Sets the hidden node architecture using a list. Model will have `len(hidden_sizes)` hidden layers, with each element being the size of its corresponding layer. [2, 3, 4] creates 3 hidden layers of with 2, 3, and 4 nodes respectively. Uses intelligent defaults otherwise. The more hidden layers / nodes per layer, the lower the final training error (generally), and the more computationally expensive the training process.
  - `learn_rate`: Determines how aggressively gradient descent runs (default 0.25). Setting too low will result in less progress made per iteration (and longer processing time). Setting too high may result in "overshooting" the optimum, or a divergent learning process.
  - `error_threshold`: The maximum acceptable average error of the model (default 0.05). The learning process will stop once errors are below the threshold or the `max_iters` has been reached, whichever comes first.
  - `max_iters`: The maximum # of iterations to be performed before stopping (default 10000).
  - `epsilon`: Determines range of initialization weights between (-epsilon, +epsilon). Uses intelligent defaults.
  - `reg_rate`: Governs regularization term (default 0). Setting this to a high value squashes weights on higher-order features with great prejudice, but tends the system of weights toward 0. Test different values with your dataset -- the ideal regularization rate may be on the order of magnitude of 0.0001, or single digit integers.
  - `momentum`: Determines the momentum factor used to "push" each iteration and weight adjustment (default 1.1). Should be greater than 1. This number will be multiplied to each gradient descent. ** keep? **
  - `log_progress`: Boolean value determining whether to log progress (default False).
  - `log_interval`: Numerical value (default 500). Model will log relevant stats every `log_interval` iterations.


###Linear Regression
#####Example: Training y = x + 14 function...
```python
data = [
    {'input': [1, 0], 'output': [1]},
    {'input': [0, 1], 'output': [1]},
    {'input': [0, 0], 'output': [0]},
    {'input': [1, 1], 'output': [1]},
]

options = {'momentum': 1.1, 'log_progress': True}

net = NeuralNet()
net.load_data(data)
net.train(options)

net.run([0, 1])     # 0.979
```

#####Restrictions:
Only supports output of 1 element. Does handle

######Adjustable model parameters(`options` dict in example)
  - `convergence_threshold`: Instead of using an error_threshold like in neural nets, linear regression uses a convergence_threshold.
  - `max_iters`: Same concept as in neural net (default 50000).
  - `learn_rate`: Same concept as neural net (default 0.01). You shouldn't need to adjust this if you leave momentum on.
  - `increase_momentum`: Boolean value determining whether to dynamically scale learning algo (default True). Strongly recommended you keep this on. If you leave it off, you'll have to adjust the learn_rate manually.


###Logistic Regression

#####Restrictions:
Only supports output of 1 element, of the value between 0 and 1. Will support multi-class in the future.

######Adjustable model parameters (these can be set similar to above)
Same API as linear regression, hallelujah.


###TODOs:
- Pruning algo for neural network to "trim" redundant nodes
- Serialization of weights, allowing user to save and resume work on large data sets
- Prettify multi-class learning for ANN by vectorizing user-given output number into 1s and 0s.
- Add features normalization
- Implement neural network momentum scaling
- Regularization in logistic and linear regression, not only ANN
- More extensive error handling -- empty layers for NN, edge-case inputs
