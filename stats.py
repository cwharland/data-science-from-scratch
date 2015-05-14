from __future__ import division
from lin_alg import *
from collections import Counter

def mean(x):
    return sum(x) / len(x)

def median(v):
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2
    
    # gotta handle the even case and odd case
    if n%2 == 1:
        return sorted_v[midpoint]
    else:
        lo = midpoint - 1
        hi = midpoint
        return (sorted_v[lo] + sorted_v[hi]) / 2

def quantile(x, p):
    '''return the pth percentile in x'''
    p_index = int(p*len(x))
    return sorted(x)[p_index]

def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i,count in counts.iteritems()
           if count == max_count]

def data_range(x):
    return max(x) - min(x)

def de_mean(x):
    m = mean(x)
    return [x_i - m
            for x_i in x]
    
def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviation(x):
    return math.sqrt(variance(x))

def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n-1)

def correlation(x, y):
    std_x = standard_deviation(x)
    std_y = standard_deviation(y)
    if std_x > 0 and std_y > 0:
        return covariance(x, y) / std_x / std_y
    else:
        return 0 # Handles the div by zero case