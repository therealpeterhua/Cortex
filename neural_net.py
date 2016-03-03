from pdb import set_trace
from math import log
from itertools import izip
from random import uniform

import utils

'''
Pruning algo -- disregard nodes that receive low weights
'''

#http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

class NeuralNet(object):
    @staticmethod
    def add_bias(input_row):
        input_row.insert(0, 1)

    def __init__(self, data = None):
        if data is not None:
            self.load_training_data(data)

    #PH:*** worth zipping these up always?
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

    def load_training_data(self, data):
        utils.prevalidate(data)
        self.training_data = utils.standardize(data)
        self.set_defaults()

    def set_defaults(self):         # defaults that can be overridden at runtime
        self.add_training_bias()
        self.reg_rate = 0.0
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
        print self.nodes
        self.weights = self.build_weights()
        self.gradients = utils.dupe_with_infs(self.weights)

        # structure same as nodes, without bias
        self.deltas = utils.dupe_with_infs(self.nodes)

        #PH:*** delete this...
        self.weights = [[[0.36483230176638926, -0.10354547213358176, -0.7980308463608701], [0.9237207276836605, -0.21646020063063934, -0.23030769700312906]], [[0.6545288228362074, 0.9406124797474309, 0.08076250834392552]]]

        # On choosing epsilons for random initialization...
        # One effective strategy for choosing epsilon is to base it on the number of units in the network. A good choice of is... LOOK UP

        #PH:*** redo the threshold here. you're trying to CONVERGE. not REACH ZERO ERROR
        self.learn_rate = 0.1               # learn rate   PH:*** rename?
        self.max_iters = 100                 # change

        # figure out right # hidden layer...
        # self.hidden_layer = max(2, self.suggested_hidden_layers())

    def add_training_bias(self):
        for input_row in self.input_iter:
            NeuralNet.add_bias(input_row)

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
        weights = [ [ [ uniform(-self.epsilon, self.epsilon)
                        for curr_node_i in xrange(curr_size) ]
                           #PH: maybe extrap out the size index picker?
                           # should only go to next sizes - 1 in next_node_i, because we don't use the current nodes to calculate BIAS unit of next layer, unless next layer is output!
                      for next_node_i in xrange(self.sizes[i + 1] - 1) ]
                    for i, curr_size in enumerate(self.sizes[:-2]) ]

        # For final output weights, we don't skip the bias unit in the next_node_i
        output_weights = [ [ uniform(-self.epsilon, self.epsilon)
                             for pre_output_node_i in xrange(self.sizes[-2]) ]
                           for output_node_i in xrange(self.sizes[-1]) ]
        weights.append(output_weights)
        return weights

    def train(self):
        iters = 0
        #PH:*** add more logic here
        while iters < self.max_iters:
            self.reset_gradients()          #PH:*** don't do this for every row!

            for input_row, output_row in self.training_iter:
                self.reset_deltas()

                self.feed_forward(input_row)
                self.back_prop(output_row)      #PH:*** you'll need to accum errors here somehow. can't just back_prop line by line...

            self.postprocess_gradients()
            self.set_new_weights()          #PH:*** calc here, and test
            iters += 1
            self.log_things()

    def log_things(self):
        new_error = self.calc_error()
        print 'Nodes: %s' % self.nodes
        print 'New error: %s' % new_error
        print 'Weights: %s' % self.weights
        print 'Gradients: %s' % self.gradients
        print 'Deltas: %s' % self.deltas
        print '\n\n'

    def reset_gradients(self):
        #PH: fix this. duping don't make sense. how about fill_with_zeros instead?
        self.gradients = utils.dupe_with_zeros(self.gradients)

    def reset_deltas(self):
        #PH: fix this. duping don't make sense. how about fill_with_zeros instead?
        #PH: is it RIGHT to always reset deltas on EVERY ROW? think so...
        self.deltas = utils.dupe_with_zeros(self.deltas)

    def run(self, user_given_input_row):
        NeuralNet.add_bias(user_given_input_row)

        self.feed_forward(user_given_input_row)
        return self.output_nodes

    #PH: extrapolate out some of these methods so you don't gotta worry about +1, -1, etc.
    def feed_forward(self, input_row):
        self.nodes[0] = input_row

        # skip the input layer
        for layer_i in xrange(self.output_layer_i):
            curr_layer_nodes = self.nodes[layer_i]
            next_layer_nodes = self.nodes[layer_i + 1]

            # multiply prev layer's nodes by next weights
            layer_weights = self.weights[layer_i]

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
        # print 'After forward feed: %s' % self.nodes

    def back_prop(self, output_row):
        self.set_deltas(output_row)
        self.accumulate_gradient(output_row)

    def accumulate_gradient(self, output_row):
        # REM: self.weights is exactly as formularized -- the first weight does indeed connect the first and second layers, and so on. len(self.weights) would be the output layer!

        for layer_i in xrange(self.output_layer_i, 0, -1):
            prev_layer_i = layer_i - 1

            curr_deltas = self.deltas[layer_i]
            prev_gradients = self.gradients[prev_layer_i]
            prev_nodes = self.nodes[prev_layer_i]

            # print 'delta rows %s columns %s' % (len(self.deltas), len(self.))

            # if prev_layer_i == 0:
            #     set_trace()

            self.fill_dot_product(curr_deltas, prev_nodes, prev_gradients)

    def fill_dot_product(self, curr_deltas, prev_nodes, prev_gradients): #tested
        for i, delta_row in enumerate(curr_deltas):
            for j, node_val in enumerate(prev_nodes):
                # print 'i, j is %s' % i, j
                # if (i, j) == (0, 3):
                #     set_trace()
                prev_gradients[i][j] += delta_row[0] * prev_nodes[j]

    def set_deltas(self, output_row):
        self.deltas[-1] = [ [predicted - actual]
                            for predicted, actual
                            in izip(self.output_nodes, output_row) ]

        # NOTE: our thetas are laid out exactly like the vectorized version -- a sub-array is a layer, then a row. We drop the first el in the row (bias), just like we drop the first column in vectorized version
        for curr_layer in xrange(len(self.deltas) - 1, 0, -1):
            prev_layer = curr_layer - 1

            if curr_layer == len(self.deltas):
                curr_deltas = last_delta
            else:
                curr_deltas = self.deltas[curr_layer]

            self.deltas[prev_layer] = self.calc_deltas(curr_deltas, prev_layer)

        self.deltas[0] = None           # PH: don't care about inputs

    def calc_deltas(self, curr_deltas, prev_layer):
        prev_weights = self.weights[prev_layer]

        # REM: curr_deltas is a column vector, of form [[5], [1], [3], [-1]]
        return (         # dot-product - check out idx reversal, on prev_weights
            [ [ sum(curr_deltas[j][0] * weights[i] * self.barbell(prev_layer, i)
                for j, weights in enumerate(prev_weights) ) ]
              for i in xrange(1, len(prev_weights[0])) ]
        )

    def barbell(self, layer, node_i):
        val = utils.sigmoid(self.nodes[layer][node_i])
        return val * (1 - val)

    def postprocess_gradients(self):
        for l, layer_gradients in enumerate(self.gradients):
            for i, next_i_gradients in enumerate(layer_gradients):
                for j, gradient in enumerate(next_i_gradients):
                    gradient_val = gradient / self.data_size
                    if j:       # j != 0
                        gradient_val += self.reg_rate * self.weights[l][i][j]
                    next_i_gradients[j] = gradient_val

    # FIX FUNCTION ORDER!
    def set_new_weights(self):
        #PH: implement momentum in a while loop here. Implement new_weights, check their errors... if higher, decrease learning rate
        for l, layer_weights in enumerate(self.weights):
            for i, next_i_weights in enumerate(layer_weights):
                for j, weight in enumerate(next_i_weights):
                    gradient_val = self.gradients[l][i][j]
                    next_i_weights[j] = weight - self.learn_rate * gradient_val

    def activate(self, layer_i):
        layer_nodes = self.nodes[layer_i]
        for i, node_val in enumerate(layer_nodes):
            # skip bias node of all but output layer (leave others as 1)
            if i == 0 and layer_i != self.output_layer_i:
                continue
            self.nodes[layer_i][i] = utils.sigmoid(node_val)

    # I assume you use this after you run all the inputs and have the predicted outputs for every one?
    # Really, you need to do this one as you're looping through and calculating the output. In a neural net, the network itself IS the hypothesis Fn... remember that calc_error is the only place we ever call the hypothesis Fn from? Well, you're gonna need to do that now with the network... probably.
    # REM the nodes don't have to be refreshed in between runs
    def calc_error(self):
        total_error = 0

        #PH:*** my regularization terms are way too strong!! Giving 300 error with the CORRECT answer, holy crap. I need it to reach to 0 with
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
            for layer in self.weights
            for next_layer_row in layer
            # don't want to add bias of curr_layer into reg term
            for i, curr_layer_weight in enumerate(next_layer_row)
        )

        return sum(all_weights_iter) * self.reg_rate / (2 * self.data_size)


net = NeuralNet([
    {'input': [1, 0], 'output': [1]},
    {'input': [0, 1], 'output': [1]},
    {'input': [0, 0], 'output': [0]},
    {'input': [1, 1], 'output': [0]},
])

print 'Before training .......'
print net.run([1, 0])
print net.run([0, 1])
print net.run([1, 1])
print net.run([0, 0])

net.train()

print 'After training .......'
print net.run([1, 0])
print net.run([0, 1])
print net.run([1, 1])
print net.run([0, 0])



#PH:*** DELETE all this stuff down here

'''
Top level of weights is level.
Second level of weights is the weights for the node of the next layer.
Third level are the individual weights to multiply the current node by.

Ie. in XOR...

(NOT AND) AND (OR)
NOT AND -->

        # self.weights = [
        #                 [[30, -20, -20], [-10, 20, 20]],       # [NOT AND], [OR]
        #                 [[-30, 20, 20]]                         # [AND]
        #                ]

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



def calc_deltas(curr_deltas, prev_weights):
    prev_deltas = []
    # more list comps here? Or break it out?
    for i in xrange(1, len(prev_weights[0])):    # could do with self.sizes?
        total = sum( curr_deltas[j][0] * weights[i]
                     for j, weights in enumerate(prev_weights) )
        prev_deltas.append([total])

    return prev_deltas


curr_deltas = [[2], [1]]
prev_weights = [[1, 2, 3, 4], [4, 3, 2, 1]]

print calc_deltas(curr_deltas, prev_weights)
