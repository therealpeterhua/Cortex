from pdb import set_trace
from itertools import izip

import validator as vl
import utils as ut


class LnrReg(object):
    def __init__(self, data = None):
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
        self.add_training_bias()

    def set_options(self, options):
        used_properties = self.get_defaults()
        ut.replace_in(used_properties, options)

        for prop_name, value in used_properties.iteritems():
            setattr(self, prop_name, value)

        self.error = float('inf')
        self.thetas = []
        self.build_thetas()

    def get_defaults(self):
        return {
            'learn_rate': 0.01,
            'threshold': 0.00001,
            'max_iters': 50000,
            'use_driver': True,
            'quick_factor': 1.1,
            'brake_factor': 0.6,
            'log_progress': True,
            'log_interval': 2000
        }

    def build_thetas(self):
        self.thetas = [0 for _ in self.training_data[0]['input']]

    def add_training_bias(self):
        for item in self.training_data:
            self.add_bias(item['input'])

    def add_bias(self, input_row):
        input_row.insert(0, 1)

    def train(self, options = None):
        if options is None:
            options = {}
        self.set_options(options)

        self.gradient_descent()

    def run(self, input_row):
        self.add_bias(input_row)
        return self.calc_hypothesis(input_row)

    def gradient_descent(self):
        iters = 0

        has_converged = False
        while (not has_converged and iters < self.max_iters):
            old_error = self.error

            if self.log_progress and not iters % self.log_interval:
                self.log_info(iters)
            self.calc_new_thetas()
            iters += 1

            has_converged = False
            if self.error == old_error:
                continue
            elif abs(old_error - self.error) < self.threshold:
                has_converged = True

        self.log_finish(iters)

    # calculate and assign new weights, else revert to old ones
    def calc_new_thetas(self):
        old_thetas, old_error = self.thetas, self.error

        partials = self.build_partials()
        new_thetas = [ curr_theta - self.learn_rate * partial_term
                        for partial_term, curr_theta
                        in izip(partials, self.thetas) ]

        self.thetas = new_thetas
        self.error = self.calc_error()

        # lower learn rate, and revert thetas and errors if grad descent fails to reduce error
        if self.use_driver and self.error < old_error:
            self.learn_rate *= self.quick_factor
        elif self.error > old_error:
            self.use_driver = False      # prevent scaling momentum up further
            self.learn_rate *= self.brake_factor
            self.thetas, self.error = old_thetas, old_error

    def build_partials(self):
        partials = []

        for j, theta in enumerate(self.thetas):
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
                    in izip(self.thetas, input_row) )

    def calc_error(self):
        return self.avg_squared_error()

    def avg_squared_error(self):
        squared_err_sum = 0
        zipped_data_iters = izip(self.input_iter, self.output_iter)

        for i, (input_row, output_val) in enumerate(zipped_data_iters):
            actual_val = output_val
            predicted_val = self.calc_hypothesis(input_row)
            squared_err_sum += (predicted_val - actual_val) ** 2

        return squared_err_sum / (2 * len(self.training_data))

    def log_info(self, num_iterations):
        dividing_line = '--------------------------------------'
        print('After %s iterations...\nThetas are %s\nError is %s\n%s' % (
            num_iterations, self.thetas, self.error, dividing_line
        ))

    def log_finish(self, iters):
        print('======================================')
        print('Finished regression in %s iterations') % iters
        print('Final error is %s') % self.error
        print('Final thetas are %s') % self.thetas
