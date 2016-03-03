from math import e
from random import uniform
from itertools import izip

def sigmoid(val):
    return 1 / (1 + e ** (-val))

def dupe_with_zeros(matrix):
    if type(matrix[0]) is list:
        return [ dupe_with_zeros(el) for el in matrix ]
    else:
        return [0 for el in matrix]

# delete? assumes matrix of consistent nesting. recursively fills with random numbers
def dupe_with_infs(matrix):
    if type(matrix[0]) is list:
        return [ dupe_with_infs(el) for el in matrix ]
    else:
        return [float('inf') for el in matrix]

# returns the dot product of 2 matrices, filling in a result matrix if given
def fill_dot_product(aye, bee, fill_in = None):
    result = ( [] if fill_in is None else fill_in )

    for i, aye_row in enumerate(aye):
        if fill_in is None:
            result.append([])
            curr_result_row = result[i]

        for j, _ in enumerate(bee[0]):
            col_iter = (bee_row[j] for bee_row in bee)
            val = sum( row_val * col_val
                       for row_val, col_val
                       in izip(aye_row, col_iter) )
            if fill_in is None:
                curr_result_row.append(val)
            else:
                result[i][j] += val

    return result

# # PH: tests, take out
# fill_in = [[None, None], [None, None], [None, None]]
# print fill_dot_product([[3], [2], [1]], [[2, 3]], fill_in)
#
# aye = [[2, 3, 4], [5, 6, 7]]
# bee = [[4, 7], [5, 8], [6, 9]]
# print fill_dot_product(aye, bee)


# The goal of this method is to raise informative errors where appropriate, ie. inconsistent data types or data lengths.
def prevalidate(data):
    if not data:
        raise Exception("Can't use blank data.")

    type_error_msg = 'Data must be a list of either lists or dicts'
    if (type(data) is not dict) and (type(data) is not list):
        raise Exception(type_error_msg)

    first_item = data[0]
    data_type = type(first_item)

    if data_type is list:
        list_length = len(first_item)
        _validate_list(data, list_length)

    elif data_type is dict:
        input_length = len(first_item['input'])
        _validate_dict(data, input_length)

    else:
        raise Exception(type_error_msg)

def standardize(data):
    if type(data[0]) is not list:
        return data

    for row_i, list_item in enumerate(data):
        output_idx = len(list_item) - 1
        data[row_i] = {
            'input': list_item[0:output_idx],
            'output': list_item[output_idx]
        }

    return data

#PH:*** change to implementation of `any`. DRY UP THE BELOW
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
def _validate_dict(data, input_length):
    data_type = dict
    for item in data:
        item_type, item_input_len = type(item), len(item['input'])
        if type(item) != data_type:
            raise Exception('Expecting %s, got %s') % (data_type, item_type)
        if item_input_len != input_length:
            error_message = 'Expecting input length of %s. Instead, got %s.' % (
                input_length, item_input_len
            )
            raise Exception(error_message)
