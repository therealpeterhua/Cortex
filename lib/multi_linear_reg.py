from pdb import set_trace
from itertools import izip

import validator as vl
import utils as ut


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
        self.add_training_bias()

    def set_options(self, options):         # defaults that can be overridden at runtime
        used_properties = self.get_defaults()
        ut.replace_in(used_properties, options)
        
        for prop_name, value in used_properties.iteritems():
            setattr(self, prop_name, value)

        self.current_error = float('inf')
        self._thetas = []
        self.build_thetas()
    
    def get_defaults(self):
        return {
            'log_progress': True,
            'log_interval': 2000,
            # Following 3 attributes govern the use of momentum in dynamically
            # adjusting the learning rate. Setting 'use_driver' to False
            # prevents the use of the quick_factor to speed up gradient descent
            'use_driver': True,
            'quick_factor': 1.1,
            'brake_factor': 0.6,
            'learn_rate': 0.01,
            'threshold': 0.00001,
            'max_iters': 50000
        }

    def build_thetas(self):
        self._thetas = [0 for _ in self.training_data[0]['input']]

    def add_training_bias(self):
        for item in self.training_data:
            self.add_bias(item['input'])

    def add_bias(self, input_row):
        input_row.insert(0, 1)

    def train(self, options = None):          #PH:*** set options...
        if options is None:
            options = {}
        self.set_options(options)
        
        self.gradient_descent()
        print( 'Thetas: ' + str(self._thetas) )
        return self._thetas

    def run(self, input_row):
        self.add_bias(input_row)                  # add bias
        return self.calc_hypothesis(input_row)

    def gradient_descent(self):
        iterations = 0

        has_converged = False
        while (not has_converged and iterations < self.max_iters):
            old_error = self.current_error
            
            if self.log_progress and not iterations % self.log_interval:
                self.log_params(iterations)
            self.calc_new_thetas()
            iterations += 1
            
            has_converged = False
            if self.current_error == old_error:
                continue
            elif abs(old_error - self.current_error) < self.threshold:
                has_converged = True
            
        print('======================================')
        print('Finished regression in ' + str(iterations) + ' iterations')

    # calculate and assign new weights, else revert to old ones
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
            self.use_driver = False      # prevent scaling momentum up further
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
        return self.avg_squared_error()

    def avg_squared_error(self):
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
regression.train({'threshold': 0.00001})
print regression.run([10])
