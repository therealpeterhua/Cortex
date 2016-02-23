from pdb import set_trace
import data_utils

class Utils:
    # The goal of this method is to raise informative errors where possible, ie. inconsistent data types or data lengths.
    @staticmethod
    def prevalidate(data):
        if not data:
            raise Exception("Can't use blank data.")

        first_item = data[0]
        data_type = type(first_item)

        if data_type is list:
            list_length = len(first_item)
            Utils._validate_list(data, list_length)
        
        elif data_type is dict:
            input_length = len(first_item['input'])
            Utils._validate_dict(data, input_length)
        
        else:
            raise Exception('Data must be a list or dictionary.')
                    
    @staticmethod
    def vectorize(data):
        if type(data[0]) is not list:
            return

        for row_i, list_item in enumerate(data):
            output_idx = len(list_item) - 1
            data[row_i] = {
                'input': list_item[0:output_idx],
                'output': list_item[output_idx]
            }
        
        return data
    
    #PH:*** change to implementation of `any`. DRY UP THE BELOW
    @staticmethod
    def _validate_list(data, list_length):
        data_type = list
        for item in data:               
            item_type, item_length = type(item), len(item)
            if (item_type != data_type) or (item_length != list_length):
                error_message = 'Expecting %s of length %s, instead got %s of length %s.' % (
                    data_type, list_length, item_type, item_length
                )
                raise Exception(error_message)
    
    #PH:*** change to implementation of `any`. DRY UP THE BELOW
    @staticmethod
    def _validate_dict(data, input_length):
        data_type = dict
        for item in data:
            item_type, item_input_len, item_output_len = (
                type(item), len(item['input']), len(item['output'])
            )
            if type(item) != data_type:
                raise Exception('Expecting %s, got %s') % (data_type, item_type)
            if item_input_len != input_length or item_output_len != 1:
                set_trace()
                error_message = 'Expecting input and output lengths of %s and %s, respectively. Instead, got %s and %s.' % (
                    input_length, 1, item_input_len, item_output_len
                )
                raise Exception(error_message)

class MultiReg:
    ROUNDING = 5
    MAX_ALLOWED_OUTPUT = 1

    def __init__(self, data = None):
        if data is not None:
            self.load_training_data(data)
        
    @property
    def thetas():
        return self.thetas

    def load_training_data(self, data):
        Utils.prevalidate(data)
        self.training_data = Utils.vectorize(data)
        self.set_defaults()

    def set_defaults(self):         # defaults that can be overridden at runtime
        self.log = False
        
        self.upward_momentum = 1.1
        self.downward_momentum = 0.70
        self.increment_upward_momentum = True   #PH:*** flip this off after first downward revision...
        
        self.add_bias()

        self.thetas = []
        self.cost_error = float('inf')
        self.build_thetas()

        #PH:*** redo the threshold here. you're trying to CONVERGE. not REACH ZERO ERROR
        self.alpha = 0.01               # learn rate   PH:*** rename?
        self.threshold = 0.00000001     # max acceptable AMSE
        self.max_iters = 40000          # make a fn of size of dataset?

    def build_thetas(self):
        self.thetas = [0 for _ in self.training_data[0]['input']]

    def add_bias(self):
        for item in self.training_data:
            item['input'].insert(0, 1)

    def output_iter(self):
        return (item['output'] for item in self.training_data)

    def input_iter(self):
        return (item['input'] for item in self.training_data)

    def run(self, options = None):          #PH:*** set options...
        self.gradient_descent()
        print( 'Thetas: ' + str(self.thetas) )
        return self.thetas

    def gradient_descent(self):
        iterations = 0

        while (self.calc_cost() > self.threshold and iterations < self.max_iters):
            if self.log:
                self.log_params()
            self.calc_new_thetas()
            iterations += 1

        print('======================================')
        print('Finished regression in ' + str(iterations) + ' iterations')

    def calc_new_thetas(self):
        partials = self.build_partials()
        # new_cost_error = self.calc_cost(new_cost_error)         #PH:*** implement this, need to test on new thetas. Will need to alter hypothesis calculation to use NEW thetas. try to use an iterator?
        
        if new_cost_error > self.cost_error and self.increase_momentum:
            self.alpha *= 
        for j, curr_theta in enumerate(self.thetas):
            self.thetas[j] = curr_theta - self.alpha * partials[j]

    # REM: can't use generator -- don't want to alter self.thetas as we're running thru
    def build_partials(self):
        partials = []

        for j, theta in enumerate(self.thetas):
            sigma = 0

            zipped_data_iters = zip(self.input_iter(), self.output_iter())
            for i, (input_row, output) in enumerate(zipped_data_iters):
                actual_val = output
                predicted_val = self.calc_hypothesis(input_row)
                delta = predicted_val - actual_val
                sigma += delta * input_row[j]

            partials.append(sigma / len(self.training_data))

        return partials

    def calc_hypothesis(self, input_row):
        output = 0
        for i, input_val in enumerate(input_row):
            output += input_val * self.thetas[i]

        return output

    def log_params(self):
        print( 'thetas are ' + str(self.thetas) )
        print( 'errors are ' + str(self.calc_cost()) )
        print( '--------------------------------------' )
    
    def calc_cost(self):
        return self.avg_mse()

    def avg_mse(self):
        squared_err_sum = 0
        zipped_data_iters = zip(self.input_iter(), self.output_iter())

        for i, (input_row, output_val) in enumerate(zipped_data_iters):
            actual_val = output_val
            predicted_val = self.calc_hypothesis(input_row)
            squared_err_sum += (predicted_val - actual_val) ** 2

        return squared_err_sum / (2 * len(self.training_data))


# test_data = [
#     {'input': [1, 1], 'output': 1},
#     {'input': [2, 2], 'output': 2},
#     {'input': [3, 3], 'output': 3}
# ]
#
# test_data_non_vectors = [
#     [1, 1, 1],
#     [2, 2, 2],
#     [3, 3, 3]
# ]

test_data_non_vectors = [           # 13962 iterations
    [4, 0],
    [5, 4],
    [6, 8],
    [7, 12],
    [8, 16],
    [9, 20],
    [10, 24]
]


test_reg = MultiReg(test_data_non_vectors)
test_reg.run()



'''
NESTED LIST COMPREHENSION...
# REM standard syntax... [fn(el) for el in list]

PH -- EXAMPLE: concating lists
lists = [['hello'], ['world', 'foo', 'bar']]
combined = [item for sublist in lists for item in sublist]

# should be read... this is a nested FOR loop! the last item increments fastest
# literally think of this like nested for loops, ie. the first line of a for loop, but without the colon after
combined = [item (for sublist in lists) (for item in sublist)]

# convert the following to floats...
int_list = [['40', '20', '10', '30'], ['20'],['100', '100']]

# joins AND converts
floating_1 = [float(el) for sublist in int_list for el in sublist]

# doesn't join, just converts. `for sublist in int_list` gets you each sub-list. You then substitute in the floating version of each sublist with `[float(el) for el in sublist]`
floating_2 = [[float(el) for el in sublist] for sublist in int_list]
'''
