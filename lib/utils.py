from math import e
from random import uniform
from itertools import izip

def sigmoid(val):
    return 1 / (1 + e ** (-val))
    
# deep duplicates a mtrx
def deep_dup(matrix):
    new_matrix = []
    if type(matrix) is not list:
        return matrix
    
    for i, element in enumerate(matrix):
        new_matrix.append(deep_dup(element))
    
    return new_matrix

def fill_with(matrix, value):
    if type(matrix) is not list:
        return value
    
    for i, element in enumerate(matrix):
        matrix[i] = fill_with(element, value)
    
    return matrix

# replaces overlapping keys in left and right with the right value
def replace_in(left, right):
    for key in right:
        if key in left:
            left[key] = right[key]

    return left

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
