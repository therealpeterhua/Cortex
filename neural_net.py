from pdb import set_trace
from math import log, e
from itertools import izip

import utils

class NeuralNet(object):
    def __init__(data):
        if data is not None:
            self.load_training_data(data)

    def load_training_data(self, data):
        utils.prevalidate(data)
        self.training_data = utils.standardize(data)
        self.set_defaults()

    def set_defaults(self):         # defaults that can be overridden at runtime
        self.log = True
        self.log_interval = 2000
        self.hidden_layers = 3

        # Following 3 attributes govern the use of momentum in dynamically
        # adjusting the learning rate. Setting self.increase_momentum to False
        # prevents the use of the quick_factor to speed up gradient descent
        self.increase_momentum = True
        self.quick_factor = 1.1
        self.brake_factor = 0.6

        self.add_bias()

        self.current_error = float('inf')
        self.theta_mtrx = []
        self.build_thetas()

        #PH:*** redo the threshold here. you're trying to CONVERGE. not REACH ZERO ERROR
        self.alpha = 0.01               # learn rate   PH:*** rename?
        self.threshold = 0.00000001     # max acceptable AMSE
        self.max_iters = 50000          # make a fn of size of dataset?

    def build_thetas():
        pass

    def train():
        pass


    def __init__(self, data = None):
        if data is not None:
            self.load_training_data(data)

    def load_training_data(self, data):
        utils.prevalidate(data)
        self.training_data = utils.standardize(data)
        self.set_defaults()

    def set_defaults(self):         # defaults that can be overridden at runtime
        self.log = True
        self.log_interval = 2000

        # Following 3 attributes govern the use of momentum in dynamically
        # adjusting the learning rate. Setting self.increase_momentum to False
        # prevents the use of the quick_factor to speed up gradient descent
        self.increase_momentum = True
        self.quick_factor = 1.1
        self.brake_factor = 0.6

        self.add_bias()

        self.current_error = float('inf')
        self.thetas = []
        self.build_thetas()

        #PH:*** redo the threshold here. you're trying to CONVERGE. not REACH ZERO ERROR
        self.alpha = 0.01               # learn rate   PH:*** rename?
        self.threshold = 0.00000001     # max acceptable AMSE
        self.max_iters = 50000          # make a fn of size of dataset?

    def build_thetas(self):
        self.thetas = [0 for _ in self.training_data[0]['input']]

    def add_bias(self):
        for item in self.training_data:
            item['input'].insert(0, 1)

    def output_iter(self):
        return (item['output'] for item in self.training_data)

    def input_iter(self):
        return (item['input'] for item in self.training_data)

    def train(self, options = None):          #PH:*** set options...
        self.gradient_descent()
        print( 'Thetas: ' + str(self.thetas) )
        return self.thetas

    def run(self, input_row):
        return self.calc_hypothesis(input_row)

    def gradient_descent(self):
        iterations = 0

        while (self.current_error > self.threshold and iterations < self.max_iters):
            if self.log and not iterations % self.log_interval:
                self.log_params(iterations)
            self.calc_new_thetas()
            iterations += 1

        print('======================================')
        print('Finished regression in ' + str(iterations) + ' iterations')

    def calc_new_thetas(self):
        old_thetas, old_error = self.thetas, self.current_error

        partials = self.build_partials()
        new_thetas = [ curr_theta - self.alpha * partial_term
                        for partial_term, curr_theta
                        in izip(partials, self.thetas) ]

        self.thetas = new_thetas
        self.current_error = self.calc_error()

        # lower learn rate, and revert thetas and errors if grad descent fails to reduce error
        if self.increase_momentum and self.current_error < old_error:
            self.alpha *= self.quick_factor
        elif self.current_error > old_error:
            self.increase_momentum = False      # prevent further scaling momentum up
            self.alpha *= self.brake_factor
            self.thetas, self.current_error = old_thetas, old_error

    def build_partials(self):
        partials = []

        for j, theta in enumerate(self.thetas):
            sigma = 0

            zipped_data_iters = izip(self.input_iter(), self.output_iter())
            for i, (input_row, output) in enumerate(zipped_data_iters):
                actual_val = output
                predicted_val = self.calc_hypothesis(input_row)
                delta = predicted_val - actual_val
                sigma += delta * input_row[j]

            partials.append(sigma / len(self.training_data))

        return partials

    def calc_hypothesis(self, input_row):
        return self.theta_trans_X(input_row)

    def theta_trans_X(self, input_row):
        return sum( theta * input_val
                    for theta, input_val
                    in izip(self.thetas, input_row) )

    def log_params(self, num_iterations):
        dividing_line = '--------------------------------------'
        print('After %s iterations...\nThetas are %s\nError is %s\n%s' % (
            num_iterations, self.thetas, self.current_error, dividing_line
        ))

    def calc_error(self):
        return self.avg_mse()

    def avg_mse(self):
        squared_err_sum = 0
        zipped_data_iters = izip(self.input_iter(), self.output_iter())

        for i, (input_row, output_val) in enumerate(zipped_data_iters):
            actual_val = output_val
            predicted_val = self.calc_hypothesis(input_row)
            squared_err_sum += (predicted_val - actual_val) ** 2

        return squared_err_sum / (2 * len(self.training_data))
