from __future__ import division
import math

def vector_add(v, w):
    '''adds two vectors element by element'''
    return [v_i + w_i
           for v_i, w_i in zip(v,w)]

def vector_subtract(v, w):
    '''subtracts element by element'''
    return [v_i - w_i
           for v_i, w_i in zip(v,w)]

# we can sum an arbitrary number of vectors with reduce
def vector_sum(vectors):
    return reduce(vector_add, vectors) # combine vectors with rule vector_add

# we can sum an arbitrary number of vectors with reduce
def vector_sum(vectors):
    return reduce(vector_add, vectors) # combine vectors with rule vector_add

def scalar_multiply(c, v):
    '''Takes scalar c and multiplies it by vector v'''
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    '''elementwise mean of a collection of vectors'''
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v, w):
    '''We want the sum of the element wise products'''
    return sum(v_i * w_i
               for v_i,w_i in zip(v,w))


def sum_of_squares(v):
    '''Simply the self dot product'''
    return dot(v,v)

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

# This is the sum of squares of the difference vector
def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

# More directly, the magnitude of the difference vector
def distance(v,w):
    return magnitude(vector_subtract(v, w))

# Shape
def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j]
           for A_i in A]
           
def make_matrix(num_rows, num_cols, make_fn):
    return [[make_fn(i, j)
            for j in xrange(num_cols)]
            for i in xrange(num_rows)]

def is_diagonal(i, j):
    '''1s on diag 0 otherwise'''
    return 1 if i == j else 0