CONGRATS YOU'VE FOUND ME!

This library implements 3 popular prediction / classification techniques.
Specifically, it implements a neural network, and logistic and linear regression.

Performs batch gradient descent across all techniques, and employs sigmoid activation for the NN and logistic reg.

Bold driver heuristic adjusts learning rate on the fly through a momentum factor (achieving multiples of efficiency gains in some cases), but feel free to turn off if you wish.

The regularization factor addresses overfit by squashing higher-order features with great prejudice, but be wary of setting its value too high. Regularization exhibits the same (though disproportionally smaller) "compression" effect on lower-order weights, tending the system towards 0.

Vectorized implementations have been implemented in Octave, and will be ported over to NumPy eventually. In the meantime, please enjoy some for loops and list comprehensions.
