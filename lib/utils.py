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

# replaces overlapping keys in left and right with the right value
def replace_in(left, right):
    for key in right:
        if key in left:
            left[key] = right[key]

    return left

# fills matrix of even nesting with infinites
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
