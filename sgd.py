from __future__ import division
from functools import partial
import random
from lin_alg import *

def difference_quotient(f, x, h):
    return (f(x + h) - f(x)) / h

derivative_estimate = partial(difference_quotient, square, h = 0.00001)

def partial_difference_quotient(f,  v, i, h):
    w = [v_j + (h if j == i else 0)
        for j, v_j in enumerate(v)]
    
    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h = 0.00001):
    return [partial_difference_quotient(f, v, i, h)
           for i, _ in enumerate(v)]

def step(v, direction, step_size):
    return [v_i + step_size * direction_i
           for v_i, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]

def safe(f):
    """error correction on apply"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance = 0.000001):
    
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    theta = theta_0 # starting location
    target_fn = safe(target_fn) # make  target safe
    value = target_fn(theta) # this is the value we minimize
    
    while True:
        gradient = gradient_fn(theta) # derivative at current param for ALL points
        next_thetas = [step(theta, gradient, -step_size)
                      for step_size in step_sizes] # find all possible next params
        
        # pick theta that minimizes the error function
        next_theta = min(next_thetas, key = target_fn)
        next_value = target_fn(next_theta)
        
        # stop if we reach tolerance
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

def negate(f):
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance = 0.000001):
    return minimize_batch(negate(target_fn),
                         negate_all(gradient_fn),
                         theta_0,
                         tolerance)

def in_random_order(data):
    indexes = [i for i,_ in enumerate(data)]
    random.shuffle(indexes)
    
    for i in indexes:
        yield data[i]

def minmize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0 = 0.01):
    
    data = zip(x,y)
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float('inf')
    iterations_with_no_improvement = 0
    
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i,y_i,theta) for x_i, y_i in data)
        
        if value < min_value:
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            iterations_with_no_improvement += 1
            alpha *= 0.9
            
        # This is the time saving step
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
            
    return min_theta
        
def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0 = 0.01):
    return maximize_stochastic(negate(target_fn),
                              negate_all(gradient_fn),
                              x, y, theta_0, alpha_0)




