from pdb import set_trace
from math import log, e
from itertools import izip
from utils import sigmoid

from multi_linear_reg import LnrReg

class LogReg(LnrReg):
    
    def calc_hypothesis(self, input_row):
        theta_trans_X = super(LogReg, self).calc_hypothesis(input_row)
        return sigmoid(theta_trans_X)

    def calc_error(self):
        return self.logistic_error()

    def logistic_error(self):
        total_error = 0
        zipped_data_iters = izip(self.input_iter, self.output_iter)

        for input_row, output_val in zipped_data_iters:
            predicted_output = self.calc_hypothesis(input_row)

            # predicted_output can == 0 or 1 due to rounding (math domain error)
            if predicted_output != output_val:
                total_error += (
                    -output_val * log(predicted_output) - (1 - output_val) * log(1 - predicted_output)
                )

        return total_error / len(self.training_data)

test_data = [
    [1, 0.9, 0],
    [5, 4, 0],
    [6, 1, 0],
    [8, 7, 0],
    [1, 3, 1],
    [1.1, 1.3, 1],
    [5, 6, 1],
    [6, 6.1, 1],
    [4, 4.5, 1]
]

regression = LogReg()
regression.load_data(test_data)
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
