from math import log
from itertools import izip
from random import uniform

from . import utils as ut
from . import validator as vl


class NeuralNet(object):

    @staticmethod
    def add_bias(input_row):
        input_row.insert(0, 1)

    def __init__(self, data = None):
        if data is not None:
            self.load_data(data)

    @property
    def input_iter(self):
        return (item['input'] for item in self.training_data)

    @property
    def output_iter(self):
        return (item['output'] for item in self.training_data)

    @property
    def training_iter(self):
        return izip(self.input_iter, self.output_iter)

    @property
    def output_nodes(self):
        return self.nodes[-1]

    def load_data(self, data):
        vl.prevalidate(data)
        self.training_data = vl.standardize(data)
        self.add_training_bias()

        self.data_size = len(self.training_data)
        self.input_size = len(self.training_data[0]['input'])
        self.output_size = len(self.training_data[0]['output'])

    def add_training_bias(self):
        for input_row in self.input_iter:
            NeuralNet.add_bias(input_row)

    def set_options(self, options):
        used_properties = self.get_defaults()

        # only allow overriding of existing defaults, forbid setting new props
        ut.replace_in(used_properties, options)

        for prop_name, value in used_properties.iteritems():
            setattr(self, prop_name, value)

        self.hidden_sizes = [size + 1 for size in self.hidden_sizes]
        self.sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        self.nodes = self.build_nodes()
        self.weights = self.build_weights()
        self.output_layer_i = len(self.nodes) - 1

        self.gradients = ut.deep_dup(self.weights)
        ut.fill_with(self.gradients, 0)

        self.deltas = ut.deep_dup(self.nodes)
        self.total_error = float('inf')

    # provides defaults dict of model params intended for user setting
    def get_defaults(self):
        hiddens = self.recommended_hidden_sizes()

        return {
            'hidden_sizes': hiddens,
            'reg_rate': 0.0,
            'epsilon': 2,
            'momentum': 0.1,
            'log_progress': False,
            'log_interval': 1000,
            'learn_rate': 0.25,
            'max_iters': 10000,
            'error_threshold': 0.05
        }

    def recommended_hidden_sizes(self):
        return [ max(4, 1 + abs(self.input_size - self.output_size) / 2) ]

    def build_nodes(self):
        nodes = [[None for i in xrange(size)] for size in self.sizes]

        # set bias values in non-output layers
        for layer_i in xrange(0, len(nodes) - 1):
            nodes[layer_i][0] = 1

        return nodes

    def build_weights(self):
        weights = [ [ [ uniform(-self.epsilon, self.epsilon)
                        for curr_node_i in xrange(curr_size) ]
                      for next_node_i in xrange(self.sizes[i + 1] - 1) ]
                    for i, curr_size in enumerate(self.sizes[:-2]) ]

        output_weights = [ [ uniform(-self.epsilon, self.epsilon)
                             for pre_output_node_i in xrange(self.sizes[-2]) ]
                           for output_node_i in xrange(self.sizes[-1]) ]
        weights.append(output_weights)
        return weights

    def run(self, user_given_input_row):
        NeuralNet.add_bias(user_given_input_row)

        self.feed_forward(user_given_input_row)
        return self.output_nodes

    def train(self, options = None):
        if options is None:
            options = {}

        self.set_options(options)
        self.log_start()
        iters = 0
        while ( iters < self.max_iters and
                self.total_error > self.error_threshold):

            old_gradient = ut.deep_dup(self.gradients)
            self.reset_gradients()

            for input_row, output_row in self.training_iter:
                self.reset_deltas()

                self.feed_forward(input_row)
                self.back_prop(output_row)

            self.postprocess_gradients()
            self.set_new_weights(old_gradient)
            iters += 1

            if self.log_progress and not iters % self.log_interval:
                self.log_info(iters)

        self.log_finish(iters)

    def reset_gradients(self):
        self.gradients = ut.fill_with(self.gradients, 0)

    def reset_deltas(self):
        self.deltas = ut.fill_with(self.deltas, 0)

    def feed_forward(self, input_row):
        self.nodes[0] = input_row

        for layer_i in xrange(self.output_layer_i):
            curr_layer_nodes = self.nodes[layer_i]
            next_layer_nodes = self.nodes[layer_i + 1]

            layer_weights = self.weights[layer_i]

            # skip calcing the bias unit, except on the output layer
            next_start_i = ( 0 if layer_i + 1 == self.output_layer_i else 1 )

            for next_node_i in xrange(next_start_i, len(next_layer_nodes)):
                weights_idx = next_node_i - 1
                next_node_weights = layer_weights[weights_idx]
                node_val = sum( curr_node_val * weight
                                for curr_node_val, weight
                                in izip(curr_layer_nodes, next_node_weights) )

                next_layer_nodes[next_node_i] = node_val

            self.activate(layer_i + 1)

    def activate(self, layer_i):
        layer_nodes = self.nodes[layer_i]
        for i, node_val in enumerate(layer_nodes):
            # skip bias node of all but output layer
            if i == 0 and layer_i != self.output_layer_i:
                continue
            self.nodes[layer_i][i] = ut.sigmoid(node_val)

    def back_prop(self, output_row):
        self.set_deltas(output_row)
        self.accumulate_gradients(output_row)

    def set_deltas(self, output_row):
        self.deltas[-1] = [ [predicted - actual]
                            for predicted, actual
                            in izip(self.output_nodes, output_row) ]

        for curr_layer in xrange(len(self.deltas) - 1, 0, -1):
            prev_layer = curr_layer - 1

            if curr_layer == len(self.deltas):
                curr_deltas = last_delta
            else:
                curr_deltas = self.deltas[curr_layer]

            self.deltas[prev_layer] = self.calc_deltas(curr_deltas, prev_layer)

        self.deltas[0] = None

    def calc_deltas(self, curr_deltas, prev_layer):
        prev_weights = self.weights[prev_layer]

        return (
            [ [ sum(curr_deltas[j][0] * weights[i] * self.barbell(prev_layer, i)
                for j, weights in enumerate(prev_weights) ) ]
              for i in xrange(1, len(prev_weights[0])) ]
        )

    def barbell(self, layer, node_i):
        node_value = self.nodes[layer][node_i]
        return node_value * (1 - node_value)

    def accumulate_gradients(self, output_row):
        for layer_i in xrange(self.output_layer_i, 0, -1):
            prev_layer_i = layer_i - 1

            curr_deltas = self.deltas[layer_i]
            prev_gradients = self.gradients[prev_layer_i]
            prev_nodes = self.nodes[prev_layer_i]

            self.fill_gradients(curr_deltas, prev_nodes, prev_gradients)

    def fill_gradients(self, curr_deltas, prev_nodes, prev_gradients):
        for i, delta_row in enumerate(curr_deltas):
            for j, node_val in enumerate(prev_nodes):
                prev_gradients[i][j] += delta_row[0] * prev_nodes[j]

    def postprocess_gradients(self):
        for l, layer_gradients in enumerate(self.gradients):
            for i, next_i_gradients in enumerate(layer_gradients):
                for j, gradient in enumerate(next_i_gradients):
                    gradient_val = gradient / self.data_size
                    if j != 0:
                        gradient_val += self.reg_rate * self.weights[l][i][j]
                    next_i_gradients[j] = gradient_val

        self.total_error = self.calc_error()

    def set_new_weights(self, old_gradient):
        for l, layer_weights in enumerate(self.weights):
            for i, next_i_weights in enumerate(layer_weights):
                for j, weight in enumerate(next_i_weights):
                    gradient_val = self.gradients[l][i][j]
                    momentum_val = old_gradient[l][i][j] * self.momentum

                    change = (self.learn_rate * gradient_val) + momentum_val
                    next_i_weights[j] = weight - change

    def calc_error(self):
        total_error = 0

        for input_row, output_row in self.training_iter:
            self.feed_forward(input_row)

            total_error += sum(
                -actual * log(predicted) - (1 - actual) * log(1 - predicted)
                for predicted, actual
                in izip(self.output_nodes, output_row)
            )

        reg_term = self.calc_error_regularization()

        return total_error / self.data_size + reg_term

    def calc_error_regularization(self):
        all_weights_iter = (
            curr_layer_weight ** 2 if i else 0
            for layer in self.weights
            for next_layer_row in layer
            for i, curr_layer_weight in enumerate(next_layer_row)
        )

        return sum(all_weights_iter) * self.reg_rate / (2 * self.data_size)

    def log_start(self):
        print 'Began training process with architecture...\n%s' % self.sizes
        print '...where each element is the number of nodes within a layer (including bias).'
        print '=======================================================\n'

    def log_info(self, iters):
        dividing_line = '\n---------------------------------------\n'
        print 'After %s iterations...\nTotal error: %s\nWeights: %s%s' % (
            iters, self.total_error, self.weights, dividing_line
        )

    def log_finish(self, iters):
        print '=======================================================\n'
        print 'Finished training in %s epochs.\nFinal error is %s' % (iters, self.total_error)
