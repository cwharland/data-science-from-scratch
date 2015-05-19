from __future__ import division
from stats import *
import random
from sgd import *
from probability import *

# work backwards
def predict(alpha, beta, x_i):
    return beta * x_i + alpha

def error(alpha, beta, x_i, y_i):
    """error for pair y_i and predict(x_i)"""
    
    return y_i - predict(alpha, beta, x_i)

def sum_of_squared_error(alpha, beta, x, y):
    return sum([error(alpha, beta, x_i, y_i)**2
               for x_i, y_i in zip(x,y)])

def least_squares_fit(x, y):
    """finds parameters alpha, beta that minimizes sum of square error"""
    beta = correlation(x,y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

# Metrics

def total_sum_of_squares(y):
    """total squared deviation of y from its mean"""
    return sum(v**2 for v in de_mean(y))

def r_squared(alpha, beta, x, y):
    """fraction of variantion in y captured by model"""
    return 1.0 - (sum_of_squared_error(alpha, beta, x, y) / total_sum_of_squares(y))


# Use GD
# try theta = [alpha, beta]

def squared_error(x_i, y_i, theta):
    alpha, beta = theta
    return error(alpha, beta, x_i, y_i) ** 2

def squared_error_gradient(x_i, y_i, theta):
    alpha, beta = theta
    
    return [-2 * error(alpha, beta, x_i, y_i), # alpha partial
           -2 * error(alpha, beta, x_i, y_i) * x_i] # beta partial

# look ma! more variables!
def predict(x_i, beta):
    return dot(x_i, beta)

def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)

def squared_error(x_i, y_i, beta):
    return error(x_i, y_i, beta) ** 2

def squared_error_gradient(x_i, y_i, beta):
    return [-2 * x_ij * error(x_i, y_i, beta)
           for x_ij in x_i]

def estimate_beta(x, y):
    beta_intial = [random.random() for x_i in x[0]]
    
    return minimize_stochastic(squared_error,
                               squared_error_gradient,
                               x,
                               y,
                              beta_intial,
                              0.001)

def multiple_r_squared(x, y, beta):
    sum_of_squared_errors = sum(error(x_i, y_i, beta) ** 2
                               for x_i, y_i in zip(x,y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)

def bootstrap_sample(data):
    """randomly sample len(data) elements with replacement"""
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data, stats_fn, num_samples):
    """evals stats_fn on num_samples bootstraps from data"""
    return [stats_fn(bootstrap_sample(data))
           for _ in range(num_samples)]

def estimate_sample_beta(sample):
    """sample data pairs"""
    x_sample, y_sample = zip(*sample) # this will always be awesome
    return estimate_beta(x_sample, y_sample)

def ridge_penalty(beta, alpha):
    return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x_i, y_i, beta, alpha):
    """estimates error on ridge penalty"""
    return error(x_i, y_i, beta) ** 2 + ridge_penalty(beta, alpha)

def ridge_penalty_gradient(beta, alpha):
    return [0] + [2 * alpha * beta_j for beta_j in beta[1:]]

def squared_error_ridge_gradient(x_i, y_i, beta, alpha):
    """gradient partials of ridge penalty"""
    return vector_add(squared_error_gradient(x_i, y_i, beta),
                     ridge_penalty_gradient(beta, alpha))

def estimate_beta_ridge(x, y, alpha):
    """gradient descent to fit ridge regression"""
    beta_initial = [random.random() for x_i in x[0]]
    return minimize_stochastic(partial(squared_error_ridge, alpha = alpha),
                              partial(squared_error_ridge_gradient, alpha = alpha),
                              x, y,
                              beta_initial,
                              0.001)

def lasso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])