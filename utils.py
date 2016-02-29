from math import e
from random import uniform

def sigmoid(val):
    return 1 / (1 + e ** (-val))

# assumes matrix of consistent nesting. recursively fills with random numbers
def dupe_with_randos(matrix):
    if type(matrix[0]) is list:
        return [ dupe_with_randos(el) for el in matrix ]
    else:
        return [uniform(1, -1) for el in matrix]

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
