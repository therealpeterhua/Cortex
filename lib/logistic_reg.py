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
