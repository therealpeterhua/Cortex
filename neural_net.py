from pdb import set_trace
from math import log
from itertools import izip
from random import uniform

import utils

'''
Pruning algo -- disregard nodes that receive low weights
'''

class NeuralNet(object):

    def suggested_hidden_layers():
        num_rows = len(self.training_data)
        # PH:*** build out more!
        # HOW TO CHOOSE NUMBER OF NODES?
        #http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

    def set_weights(nested_matrices):       #PH:*** used only for testing.. lol
        self.weights = nested_matrices

    def build_weights():
        pass

    def train():
        pass

    def activate():
        pass


class NeuralNet(object):
    @staticmethod
    def add_bias(input_row):
        input_row.insert(0, 1)

    def __init__(self, data = None):
        if data is not None:
            self.load_training_data(data)

    def load_training_data(self, data):
        utils.prevalidate(data)
        self.training_data = utils.standardize(data)
        self.set_defaults()

    def set_defaults(self):         # defaults that can be overridden at runtime
        self.reg_rate = 1           #PH: reasonable?
        self.epsilon = 1       #PH: look at suggested here, or revisit

        self.data_size = len(self.training_data)        #PH: confusing naming?

        self.input_size = 2
        self.input_size += 1        # add bias
        self.output_size = 1        # read from training_data

        #PH:*** have defaults, allow USER to SET hidden_sizes
        self.hidden_sizes = [2]
        # add biases
        self.hidden_sizes = [size + 1 for size in self.hidden_sizes]
        self.sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        self.build_nodes()
        self.build_weights()
        self.weights = [
                        [[30, -20, -20], [-10, 20, 20]],       # [NOT AND], [OR]
                        [[-30, 20, 20]]                         # [AND]
                       ]
        self.errors = []
        self.build_errors_matrix()

        # On choosing epsilons for random initialization...
        # One effective strategy for choosing epsilon is to base it on the number of units in the network. A good choice of is... LOOK UP

        #PH:*** redo the threshold here. you're trying to CONVERGE. not REACH ZERO ERROR
        self.alpha = 0.01               # learn rate   PH:*** rename?

        # figure out right # hidden layer...
        # self.hidden_layer = max(2, self.suggested_hidden_layers())

    def parse_and_set(self, inputs):
        # set input_size, output_size, output_layer_i, sizes?
        pass

    def build_nodes(self):
        self.nodes = [[None for i in xrange(size)] for size in self.sizes]
        #PH:*** delete if not used again
        self.output_layer_i = len(self.nodes) - 1
        # set bias values in non-output layers
        for layer_i in xrange(0, self.output_layer_i):
            self.nodes[layer_i][0] = 1

    def build_weights(self):
        # For each layer, as many rows as there are nodes in the next layer. As many elements per row as there are nodes in current layer.
        self.weights = [ [ [ uniform(-self.epsilon, self.epsilon)
                             for curr_node_i in xrange(curr_size) ]
                           #PH: maybe extrap out the size index picker?
                           # should only go to next sizes - 1 in next_node_i, because we don't use the current nodes to calculate BIAS unit of next layer, unless next layer is output!
                           for next_node_i in xrange(self.sizes[i + 1] - 1) ]
                         for i, curr_size in enumerate(self.sizes[:-2]) ]

        # For final output weights, we don't skip the bias unit in the next_node_i
        output_weights = [ [ uniform(-self.epsilon, self.epsilon)
                             for pre_output_node_i in xrange(self.sizes[-2]) ]
                           for output_node_i in xrange(self.sizes[-1]) ]
        self.weights.append(output_weights)

    def build_errors_matrix(self):
        self.errors = utils.dupe_with_randos(self.weights)

    def output_iter(self):
        return (item['output'] for item in self.training_data)

    def input_iter(self):
        return (item['input'] for item in self.training_data)

    def train(self):
        training_iters = izip(self.input_iter(), self.output_iter())
        for input_row, output_val in self.training_data:
            self.run(input_row)
            self.back_prop()

    def run(self, input_row):
        NeuralNet.add_bias(input_row)
        self.nodes[0] = input_row
        self.output_layer_i = len(self.nodes) - 1

        self.feed_forward()
        return self.nodes[self.output_layer_i]

    #PH: extrapolate out some of these methods so you don't gotta worry about +1, -1, etc.
    def feed_forward(self):
        # skip the input layer
        for layer_i in xrange(0, self.output_layer_i):
            curr_layer_nodes = self.nodes[layer_i]
            next_layer_nodes = self.nodes[layer_i + 1]

            # multiply prev layer's nodes by next weights
            layer_weights = self.weights[layer_i]

            # skip calcing the bias unit on the curr node, unless you're on the output layer -- there's no bias unit there.
            next_start_i = ( 0 if layer_i + 1 == self.output_layer_i
                                      else 1 )

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
            print self.nodes

    def back_prop(self):
        pass

    def activate(self, layer_i):
        layer_nodes = self.nodes[layer_i]
        for i, node_val in enumerate(layer_nodes):
            # don't activate bias term, just leave as a 1. Unless you're on the output layer, then you will need to activate first term
            # if layer_i == self.output_layer_i or i != 0:
            if i == 0 and layer_i != self.output_layer_i:
                continue
            self.nodes[layer_i][i] = utils.sigmoid(node_val)

    # I assume you use this after you run all the inputs and have the predicted outputs for every one?
    # Really, you need to do this one as you're looping through and calculating the output. In a neural net, the network itself IS the hypothesis Fn... remember that calc_error is the only place we ever call the hypothesis Fn from? Well, you're gonna need to do that now with the network... probably.
    # REM the nodes don't have to be refreshed in between runs
    def calc_error(self):
        self.output_size
        reg_term = self.calc_error_regularization()

    def calc_error_regularization(self):
        all_weights_iter = (
            curr_layer_weight ** 2 if i else 0      # implicit 0
            for layer in self.weights
            for next_layer_row in layer
            # don't want to add bias of curr_layer into reg term
            for i, curr_layer_weight in enumerate(next_layer_row)
        )

        return sum(all_weights_iter) * self.reg_rate / (2 * self.data_size)


net = NeuralNet([[1, 0, 0], [0, 1, 1]])
print net.run([1, 0])
print net.calc_error_regularization()



'''
Top level of weights is level.
Second level of weights is the weights for the node of the next layer.
Third level are the individual weights to multiply the current node by.

Ie. in XOR...

(NOT AND) AND (OR)
NOT AND -->
'''

# PH: Non-list-comp setting own weights
# self.other_weights = []
# for i, curr_size in enumerate(self.sizes[:-1]):
#     self.other_weights.append([])
#     level_2 = self.other_weights[i]
#
#     # -1 bc don't use curr nodes to calculate bias unit of next nodes
#     for next_node_i in xrange(self.sizes[i + 1] - 1):
#         level_2.append([])
#         level_3 = level_2[next_node_i]
#
#         for curr_node_i in xrange(curr_size):
#             level_3.append(1)

# PH: Non-list-comp regularization
# def calc_error_regularization(self):
#     reg = 0
#
#     for layer_weights in self.weights:
#         for curr_layer_weights in layer_weights:
#             for i, weight in enumerate(curr_layer_weights):
#                 if i != 0:
#                     reg += (weight ** 2)
#     return reg * self.reg_rate / (2 * self.data_size)

# For weights node sizes ....
# self.sizes = [3, 4, 5, 2]
# NOTE: the first weights will be (4 - 1) x 3, the second (5 - 1) x 4, etc.
# ^ This will be the size of the matrices

from random import randrange
from functools import partial

def random(a, b):
    return randrange(a, b)

my_func = partial(random, -5, 5)
