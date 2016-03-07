from pdb import set_trace
from itertools import izip

import validator as vl


class LnrReg(object):
    def __init__(self, data = None):
        # PH:*** TAKE OPTIONS OF PARAMETERS, ALONG WITH DATA
        # decompose in the load_data, or elsewhere, with the fn(**args)
        if data is not None:
            self.load_data(data)

    @property
    def output_iter(self):
        return (item['output'][0] for item in self.training_data)

    @property
    def input_iter(self):
        return (item['input'] for item in self.training_data)

    def load_data(self, data):
        vl.prevalidate(data)
        self.training_data = vl.standardize(data)
        self.set_defaults()

    def set_defaults(self):         # defaults that can be overridden at runtime
        self.log_progress = True
        self.log_interval = 2000

        # Following 3 attributes govern the use of momentum in dynamically
        # adjusting the learning rate. Setting self.use_driver to False
        # prevents the use of the quick_factor to speed up gradient descent
        self.use_driver = True
        self.quick_factor = 1.1
        self.brake_factor = 0.6

        self.add_training_bias()

        self.current_error = float('inf')
        self._thetas = []
        self.build_thetas()

        #PH:*** redo the threshold here. you're trying to CONVERGE. not REACH ZERO ERROR
        self.learn_rate = 0.01               # learn rate   PH:*** rename?
        self.threshold = 0.00000001     # max acceptable AMSE
        self.max_iters = 50000          # make a fn of size of dataset?

    def build_thetas(self):
        self._thetas = [0 for _ in self.training_data[0]['input']]

    def add_training_bias(self):
        for item in self.training_data:
            self.add_bias(item['input'])

    def add_bias(self, input_row):
        input_row.insert(0, 1)

    def train(self, options = None):          #PH:*** set options...
        self.gradient_descent()
        print( 'Thetas: ' + str(self._thetas) )
        return self._thetas

    def run(self, input_row):
        self.add_bias(input_row)                  # add bias
        return self.calc_hypothesis(input_row)

    def gradient_descent(self):
        iterations = 0

        while (self.current_error > self.threshold and iterations < self.max_iters):
            if self.log_progress and not iterations % self.log_interval:
                self.log_params(iterations)
            self.calc_new_thetas()
            iterations += 1

        print('======================================')
        print('Finished regression in ' + str(iterations) + ' iterations')

    def calc_new_thetas(self):
        old_thetas, old_error = self._thetas, self.current_error

        partials = self.build_partials()
        new_thetas = [ curr_theta - self.learn_rate * partial_term
                        for partial_term, curr_theta
                        in izip(partials, self._thetas) ]

        self._thetas = new_thetas
        self.current_error = self.calc_error()

        # lower learn rate, and revert thetas and errors if grad descent fails to reduce error
        if self.use_driver and self.current_error < old_error:
            self.learn_rate *= self.quick_factor
        elif self.current_error > old_error:
            self.use_driver = False      # prevent further scaling momentum up
            self.learn_rate *= self.brake_factor
            self._thetas, self.current_error = old_thetas, old_error

    def build_partials(self):
        partials = []

        for j, theta in enumerate(self._thetas):
            sigma = 0

            zipped_data_iters = izip(self.input_iter, self.output_iter)
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
                    in izip(self._thetas, input_row) )

    def log_params(self, num_iterations):
        dividing_line = '--------------------------------------'
        print('After %s iterations...\nThetas are %s\nError is %s\n%s' % (
            num_iterations, self._thetas, self.current_error, dividing_line
        ))

    def calc_error(self):
        return self.avg_mse()

    def avg_mse(self):
        squared_err_sum = 0
        zipped_data_iters = izip(self.input_iter, self.output_iter)

        for i, (input_row, output_val) in enumerate(zipped_data_iters):
            actual_val = output_val
            predicted_val = self.calc_hypothesis(input_row)
            squared_err_sum += (predicted_val - actual_val) ** 2

        return squared_err_sum / (2 * len(self.training_data))

data = [
  {'input': [4], 'output': [9]},      # [4, 0]   would also work
  {'input': [2.5], 'output': [3]},    # [2.5, 3] would also work
  {'input': [7], 'output': [21]},
  {'input': [-2], 'output': [-15]}
]

regression = LnrReg(data)
regression.train()
print regression.run([10])
