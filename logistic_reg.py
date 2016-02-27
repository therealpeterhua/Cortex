from pdb import set_trace
from math import log, e
from itertools import izip
from utils import sigmoid

from multi_linear_reg import LnrReg
'''

1) Inherit from linear reg
2) Change cost function, not mse
3) Wrap hypo in sigmoid
4) Convert to Python 2?
5) Have a way to actually GET THE PREDICTION!!
6) Create a way to serialize work done so for -- all you need are the thetas
7) Add regularization to logistic regression -- else thetas can get huge

'''

regression = LnrReg()

class LogReg(LnrReg):
    def __init__(self, data = None):
        super(LogReg, self).__init__(data)

    def set_defaults(self):
        super(LogReg, self).set_defaults()

    def calc_hypothesis(self, input_row):
        theta_trans_X = super(LogReg, self).calc_hypothesis(input_row)
        return sigmoid(theta_trans_X)

    def calc_error(self):
        return self.logistic_error()

    def logistic_error(self):
        total_error = 0
        zipped_data_iters = izip(self.input_iter(), self.output_iter())

        for input_row, output_val in zipped_data_iters:
            predicted_output = self.calc_hypothesis(input_row)

            # predicted_output can == 0 or 1 due to rounding
            if predicted_output != output_val:
                total_error += (
                    -output_val * log(predicted_output) - (1 - output_val) * log(1 - predicted_output)
                )

        return total_error / len(self.training_data)



test_data = [
    {'input': [1, 0.9], 'output': 0},
    {'input': [5, 4], 'output': 0},
    {'input': [6, 1], 'output': 0},
    {'input': [8, 7], 'output': 0},
    {'input': [1, 3], 'output': 1},
    {'input': [1.1, 1.3], 'output': 1},
    {'input': [5, 6], 'output': 1},
    {'input': [6, 6.1], 'output': 1},
    {'input': [4, 4.5], 'output': 1}
]

regression = LogReg(test_data)
regression.train()
print regression.run([1, 0.5])
print regression.run([5, 4])
print regression.run([7, 5])
print regression.run([8, 9])
print regression.run([9, 11])
print regression.run([10, 15])

# test_data = [
#     {'input': [0], 'output': 0},
#     {'input': [4], 'output': 0},
#     {'input': [4], 'output': 0},
#     {'input': [4], 'output': 0},
#     {'input': [6], 'output': 1},
#     {'input': [6], 'output': 1},
#     {'input': [6], 'output': 1},
#     {'input': [10], 'output': 1},
#     {'input': [7], 'output': 1}
# ]

# regression = LogReg(test_data)
# regression.train()
# print regression.run([1, 20])
# print regression.run([1, 10])
# print regression.run([1, 6])
# print regression.run([1, 5.5])
# print regression.run([1, 4])
# print regression.run([1, 3])
