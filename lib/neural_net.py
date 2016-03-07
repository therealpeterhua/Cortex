from pdb import set_trace
from math import log
from itertools import izip
from random import uniform

import utils as ut
import validator as vl

'''
NOTES ON XOR BATCH ANN GRADIENT DESCENT
1) Low epsilon (0.15) tends all weights toward 0.5
2) Having more nodes in hidden layer increases chance of accurate training. Bigger brainz

- Follow rest of notes (set logistic and linear regression to take [0], [1] for outputs)
- Implement error threshold for neural nets, and "convergence" theta-change threshold for logistic and linear regression
- Set intelligent defaults for node_size, # hidden layers, etc.
- Use intelligent epsilons!

- Finish readme -- each section needs its own. ANN first.
- Humanize all variable names. No thetas & alphas.
- Wrap all these in a callable format. `lib`, etc.

* Add in suggested epsilons per layer -- isn't there a simple number for this?
* Print initialization conditions (number of hidden nodes, etc.).
* Underscore the attributes you don't wish the user to set
* Get rid of momentum altogether?
* See to calculation redundancies -- ie. calcing errors twice better for logging
* Revise how you talk about 'momentum'. Either implement it correctly in neural_net, or not.
* Add overarching library import! Can you namespace directly from the module?
* Work out the bold driver here.
* Go thru and address all comments, esp. ***
* Strip underscores from property names?
* Straighten out some of the spaghetti properties (ie. output_layer_i)
* Delete all unnecessary code -- training examples at end, unused methods.
* Should probably delete the underscores on properties -- just looks messy

Linear / Logistic Regression
* Switch example to use multiple inputs
* Fill out rest of README

'''

#http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

class NeuralNet(object):
    @staticmethod
    def add_bias(input_row):
        input_row.insert(0, 1)

    def __init__(self, data = None):
        self.training_data, self.data_size = None, None

        self._nodes, self._sizes = None, None
        self.hidden_sizes, self.input_size, self.output_size = None, None, None
        self._weights, self._deltas, self._gradients = None, None, None
        self.output_layer_i = None

        self.reg_rate = None
        self.epsilon = None
        self.momentum = None
        self.learn_rate = None
        self.max_iters, self.error_threshold = None, None
        self.log_progress, self.log_interval = None, None

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
        return self._nodes[-1]

    def load_data(self, data):
        vl.prevalidate(data)
        self.training_data = vl.standardize(data)
        self.add_training_bias()

        self.data_size = len(self.training_data)        #PH: confusing naming?
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
        self._sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        self.build_nodes()
        self._weights = self.build_weights()

        self._gradients = ut.deep_dup(self._weights)
        ut.fill_with(self._gradients, float('inf'))

        # structure same as nodes, without bias
        self._deltas = ut.deep_dup(self._nodes)
        self._total_error = float('inf')

    # provides defaults dict of model params intended for user setting
    def get_defaults(self):
        hiddens = self.recommended_hidden_sizes()

        return {
            'hidden_sizes': hiddens,
            'reg_rate': 0.0,
            'epsilon': 2,
            'momentum': 1.1,
            'log_progress': False,
            'log_interval': 1000,
            'learn_rate': 0.25,
            'max_iters': 10000,
            'error_threshold': 0.05
        }

    def recommended_hidden_sizes(self):
        return [ max(4, 1 + abs(self.input_size - self.output_size) / 2) ]

    def build_nodes(self):
        self._nodes = [[None for i in xrange(size)] for size in self._sizes]
        #PH:*** delete if not used again
        self.output_layer_i = len(self._nodes) - 1
        # set bias values in non-output layers
        for layer_i in xrange(0, self.output_layer_i):
            self._nodes[layer_i][0] = 1

    def build_weights(self):
        # For each layer, as many rows as there are nodes in the next layer. As many elements per row as there are nodes in current layer.
        weights = [ [ [ uniform(-self.epsilon, self.epsilon)
                        for curr_node_i in xrange(curr_size) ]
                           #PH: maybe extrap out the size index picker?
                           # should only go to next sizes - 1 in next_node_i, because we don't use the current nodes to calculate BIAS unit of next layer, unless next layer is output!
                      for next_node_i in xrange(self._sizes[i + 1] - 1) ]
                    for i, curr_size in enumerate(self._sizes[:-2]) ]

        # For final output weights, we don't skip the bias unit in the next_node_i
        output_weights = [ [ uniform(-self.epsilon, self.epsilon)
                             for pre_output_node_i in xrange(self._sizes[-2]) ]
                           for output_node_i in xrange(self._sizes[-1]) ]
        weights.append(output_weights)
        return weights

    def train(self, options = None):
        if options is None:
            options = {}

        self.set_options(options)             #PH:***** finish this
        self.log_start()
        iters = 0
        while ( iters < self.max_iters and
                self._total_error > self.error_threshold):
            
            old_gradient = ut.deep_dup(self._gradients)
            self.reset_gradients()          #PH:*** don't do this for every row!

            for input_row, output_row in self.training_iter:
                self.reset_deltas()

                self.feed_forward(input_row)
                self.back_prop(output_row)

            self.postprocess_gradients()
            self.set_new_weights(old_gradient)          #PH:*** Bold drive here
            iters += 1

            if self.log_progress and not iters % self.log_interval:
                self.log_info(iters)

        self.log_finish(iters)

    def log_start(self):
        print 'Began training process with architecture...\n%s' % self._sizes
        print '...where each element is the number of nodes within a layer (including bias).'
        print '=======================================================\n'

    def log_info(self, iters):
        dividing_line = '\n---------------------------------------\n'
        print 'After %s iterations...\nTotal error: %s\nWeights: %s%s' % (
            iters, self._total_error, self._weights, dividing_line
        )

    def log_finish(self, iters):
        print '=======================================================\n'
        print 'Finished training in %s epochs.\nFinal error is %s' % (iters, self._total_error)

    def reset_gradients(self):
        self._gradients = ut.fill_with(self._gradients, 0)

    def reset_deltas(self):
        self._deltas = ut.fill_with(self._deltas, 0)

    def run(self, user_given_input_row):
        NeuralNet.add_bias(user_given_input_row)

        self.feed_forward(user_given_input_row)
        return self.output_nodes

    #PH: reroute these loops, or use islice, so you don't gotta worry about +1, -1, etc.
    def feed_forward(self, input_row):
        self._nodes[0] = input_row

        # skip the input layer
        for layer_i in xrange(self.output_layer_i):
            curr_layer_nodes = self._nodes[layer_i]
            next_layer_nodes = self._nodes[layer_i + 1]

            # multiply prev layer's nodes by next weights
            layer_weights = self._weights[layer_i]

            # skip calcing the bias unit on the curr node, unless you're on the output layer -- there's no bias unit there.
            next_start_i = ( 0 if layer_i + 1 == self.output_layer_i else 1 )

            # fill in next layer nodes
            for next_node_i in xrange(next_start_i, len(next_layer_nodes)):
                # optimize this later, but REM we don't keep any weights to calc the x_0 for the NEXT layer. That's always 1.
                # also, ok if there's only 1 thing in weights.. weights[0] == weights[1]
                weights_idx = next_node_i - 1
                next_node_weights = layer_weights[weights_idx]
                node_val = sum( curr_node_val * weight
                                for curr_node_val, weight
                                in izip(curr_layer_nodes, next_node_weights) )

                next_layer_nodes[next_node_i] = node_val

            self.activate(layer_i + 1)

    #PH: move closer to train() function in positioning O_O
    def back_prop(self, output_row):
        self.set_deltas(output_row)
        self.accumulate_gradient(output_row)

    def accumulate_gradient(self, output_row):
        # REM: self._weights is exactly as formularized -- the first weight does indeed connect the first and second layers, and so on. len(self._weights) would be the output layer!

        for layer_i in xrange(self.output_layer_i, 0, -1):
            prev_layer_i = layer_i - 1

            curr_deltas = self._deltas[layer_i]
            prev_gradients = self._gradients[prev_layer_i]
            prev_nodes = self._nodes[prev_layer_i]

            self.fill_gradients(curr_deltas, prev_nodes, prev_gradients)

    def fill_gradients(self, curr_deltas, prev_nodes, prev_gradients): #tested
        for i, delta_row in enumerate(curr_deltas):
            for j, node_val in enumerate(prev_nodes):
                prev_gradients[i][j] += delta_row[0] * prev_nodes[j]

    def set_deltas(self, output_row):
        self._deltas[-1] = [ [predicted - actual]
                            for predicted, actual
                            in izip(self.output_nodes, output_row) ]

        # NOTE: our thetas are laid out exactly like the vectorized version -- a sub-array is a layer, then a row. We drop the first el in the row (bias), just like we drop the first column in vectorized version
        for curr_layer in xrange(len(self._deltas) - 1, 0, -1):
            prev_layer = curr_layer - 1

            if curr_layer == len(self._deltas):
                curr_deltas = last_delta
            else:
                curr_deltas = self._deltas[curr_layer]

            self._deltas[prev_layer] = self.calc_deltas(curr_deltas, prev_layer)

        self._deltas[0] = None           # PH: don't care about inputs

    def calc_deltas(self, curr_deltas, prev_layer):
        prev_weights = self._weights[prev_layer]

        # REM: curr_deltas is a column vector, of form [[5], [1], [3], [-1]]
        return (         # dot-product - check out idx reversal, on prev_weights
            [ [ sum(curr_deltas[j][0] * weights[i] * self.barbell(prev_layer, i)
                for j, weights in enumerate(prev_weights) ) ]
              for i in xrange(1, len(prev_weights[0])) ]
        )

    def barbell(self, layer, node_i):
        #PH:*** NOTE val used to be the sigmoid! not anymore...
        node_value = self._nodes[layer][node_i]
        return node_value * (1 - node_value)

    def postprocess_gradients(self):
        for l, layer_gradients in enumerate(self._gradients):
            for i, next_i_gradients in enumerate(layer_gradients):
                for j, gradient in enumerate(next_i_gradients):
                    gradient_val = gradient / self.data_size
                    if j != 0:
                        gradient_val += self.reg_rate * self._weights[l][i][j]
                    next_i_gradients[j] = gradient_val

        self._total_error = self.calc_error()

    #PH:*** FIX FUNCTION ORDER!
    def set_new_weights(self, old_gradient):
        #PH: implement momentum in a while loop here. Implement new_weights, check their errors... if higher, decrease learning rate
        for l, layer_weights in enumerate(self._weights):
            for i, next_i_weights in enumerate(layer_weights):
                for j, weight in enumerate(next_i_weights):
                    gradient_val = self._gradients[l][i][j]
                    change = self.momentum * self.learn_rate * gradient_val
                    next_i_weights[j] = weight - change

    def activate(self, layer_i):
        layer_nodes = self._nodes[layer_i]
        for i, node_val in enumerate(layer_nodes):
            # skip bias node of all but output layer (leave others as 1)
            if i == 0 and layer_i != self.output_layer_i:
                continue
            self._nodes[layer_i][i] = ut.sigmoid(node_val)

    # I assume you use this after you run all the inputs and have the predicted outputs for every one?
    # Really, you need to do this one as you're looping through and calculating the output. In a neural net, the network itself IS the hypothesis Fn... remember that calc_error is the only place we ever call the hypothesis Fn from? Well, you're gonna need to do that now with the network... probably.
    # REM the nodes don't have to be refreshed in between runs
    def calc_error(self):
        total_error = 0

        for input_row, output_row in self.training_iter:
            self.feed_forward(input_row)             # fill nodes with the current input_row
            #PH: probably want to yield into this from the main function? So avoid running all the inputs twice, once for gradient descent and again for error calc

            total_error += sum(
                -actual * log(predicted) - (1 - actual) * log(1 - predicted)
                for predicted, actual
                in izip(self.output_nodes, output_row)
            )

        reg_term = self.calc_error_regularization()

        return total_error / self.data_size + reg_term

    def calc_error_regularization(self):
        all_weights_iter = (
            curr_layer_weight ** 2 if i else 0      # implicit 0
            for layer in self._weights
            for next_layer_row in layer
            # don't want to add bias of curr_layer into reg term
            for i, curr_layer_weight in enumerate(next_layer_row)
        )

        return sum(all_weights_iter) * self.reg_rate / (2 * self.data_size)


net = NeuralNet([
    {'input': [1, 0], 'output': [1]},
    {'input': [0, 1], 'output': [1]},
    {'input': [1, 1], 'output': [0]},
    {'input': [0, 0], 'output': [0]}
])

net.train({'hidden_sizes': [3, 3], 'log_progress': True})

print net.run([1, 0])
print net.run([0, 1])
print net.run([1, 1])
print net.run([0, 0])
