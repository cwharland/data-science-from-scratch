from __future__ import division
from wwd import *
from linear_regression import *
from ml import *

def logistic(x):
    return 1.0 / (1 + math.exp(-x))
def logistic_prime(x):
    return logistic(x) * (1 - logistic(x))
def logistic_log_likelihood_i(x_i, y_i, beta):
    if y_i == 1:
        return math.log(logistic(dot(x_i, beta)))
    if y_i == 0:
        return math.log(1 - logistic(dot(x_i, beta)))
# assume points are independent
def logistic_log_likelihood(x, y, beta):
    return sum(logistic_log_likelihood_i(x_i, y_i, beta)
               for x_i, y_i in zip(x, y))
# Gradients
def logistic_log_partial_ij(x_i, y_i, beta, j):
    """i is index of point and j is index of derivative"""
    return (y_i - logistic(dot(x_i, beta))) * x_i[j]

def logistic_log_gradient_i(x_i, y_i, beta):
    """gradient of ll for ith point"""
    return [logistic_log_partial_ij(x_i, y_i, beta, j)
           for j, _ in enumerate(beta)]

def logistic_log_gradient(x, y, beta):
    return reduce(vector_add,
                 [logistic_log_gradient_i(x_i, y_i, beta)
                 for x_i, y_i in zip(x,y)])