from pdb import set_trace
from math import log
from itertools import izip

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

    def forward_prop():
        pass

    def activate():
        pass


    # PH: below are lifted straight from linear regression


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
        self.nodes = [[1, None, None], [1, None, None], [None]]
        self.sizes = [len(layer_nodes) for layer_nodes in self.nodes]
        self.output_size = self.sizes[-1]
        self.output_layer_i = len(self.nodes) - 1
        self.weights = [
                        [None],                                 # l_input
                        [[30, -20, -20], [-10, 20, 20]],        # [NOT AND], [OR]
                        [[-30, 20, 20]]                         # [AND]
                       ]

        #PH:*** redo the threshold here. you're trying to CONVERGE. not REACH ZERO ERROR
        self.alpha = 0.01               # learn rate   PH:*** rename?

        # figure out right # hidden layer...
        # self.hidden_layer = max(2, self.suggested_hidden_layers())

    def run(self, input_row):
        NeuralNet.add_bias(input_row)
        self.nodes[0] = input_row
        self.output_layer_i = len(self.nodes) - 1

        self.forward_prop()
        return self.nodes[self.output_layer_i]

    def forward_prop(self):
        # skip the input layer
        for layer_idx in xrange(1, self.output_layer_i + 1):
            prev_layer_idx = layer_idx - 1
            curr_layer_nodes = self.nodes[layer_idx]  #PH: need this?

            # multiply prev layer's nodes by current weights
            prev_layer_nodes = self.nodes[prev_layer_idx]
            layer_weights = self.weights[layer_idx]

            # skip assigning the bias unit from prev node, unless you're on the output layer -- there's no bias unit there.
            start_curr_node_idx = (0 if layer_idx == self.output_layer_i else 1)

            # fill in current layer nodes
            for curr_node_idx in xrange(start_curr_node_idx, len(curr_layer_nodes)):
                # optimize this later, but REM we don't keep any weights to calc the x_0 for the CURRENT layer. That's always 1.
                weights_idx = curr_node_idx - 1
                curr_node_weights = layer_weights[weights_idx]
                node_val = sum( prev_node_val * weight
                                for prev_node_val, weight
                                in izip(prev_layer_nodes, curr_node_weights) )

                curr_layer_nodes[curr_node_idx] = node_val

            self.activate(layer_idx)

    def activate(self, layer_idx):
        layer_nodes = self.nodes[layer_idx]
        for i, node_val in enumerate(layer_nodes):
            self.nodes[layer_idx][i] = utils.sigmoid(node_val)

    def output_iter(self):
        return (item['output'] for item in self.training_data)

    def input_iter(self):
        return (item['input'] for item in self.training_data)

    # I assume you use this after you run all the inputs and have the predicted outputs for every one?
    # Really, you need to do this one as you're looping through and calculating the output. In a neural net, the network itself IS the hypothesis Fn... remember that calc_error is the only place we ever call the hypothesis Fn from? Well, you're gonna need to do that now with the network... probably.
    # REM the nodes don't have to be refreshed in between runs
    def calc_error(self):
        self.output_size
        reg_term = self.calc_error_regularization_term()

    #PH:*** fix this. Need to exclude bias terms.
    def calc_error_regularization_term(self):
        # oops. short variable names to avoid newlines
        layer_weights_iter = (lw for lw in self.weights)
        curr_layer_idx_weights_iter = (clw for clw in layer_weights_iter)
        weights_iter = (w ** 2 for w in curr_layer_idx_weights_iter)

        return sum(weights_iter)



net = NeuralNet([[1, 0, 0], [0, 1, 1]])
print net.run([0, 0])



'''
Top level of weights is level.
Second level of weights is the weights for the node of the next layer.
Third level are the individual weights to multiply the current node by.

Ie. in XOR...

(NOT AND) AND (OR)
NOT AND -->
'''

my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
split_list = [[1, 2], 3]]


layer_weights_iter = (lw for lw in weights)
curr_layer_idx_weights_iter = (clw for clw in layer_weights_iter)
weights_iter = (w ** 2 for w in curr_layer_idx_weights_iter)

sum(weights_iter)
