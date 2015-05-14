from __future__ import division
import math
from probability import *

def normal_approximation_to_binomial(n, p):
    mu = p*n
    sigma = math.sqrt(p * (1-p) * n)
    return mu, sigma

normal_prob_below = normal_cdf

def normal_prob_above(lo, mu = 0, sigma = 1):
    return 1 - normal_cdf(lo, mu, sigma)

def normal_prob_between(lo, hi, mu = 0, sigma = 1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def normal_prob_outside(lo, hi, mu = 0, sigma = 1):
    1 - normal_prob_between(lo, hi, mu, sigma)

def normal_upper_bound(p, mu = 0, sigma = 1):
    return inverse_normal_cdf(p, mu, sigma)

def normal_lower_bound(p, mu = 0, sigma = 1):
    return inverse_normal_cdf(1 - p, mu, sigma)

def normal_two_sided_bounds(p, mu = 0, sigma = 1):
    tail_prob = (1 - p) / 2
    
    upper_bound = normal_lower_bound(tail_prob, mu, sigma)
    lower_bound = normal_upper_bound(tail_prob, mu, sigma)
    
    return lower_bound, upper_bound

def two_sided_p_value(x, mu = 0, sigma = 1):
    if x >= mu:
        return 2 * normal_prob_above(x, mu, sigma)
    else:
        return 2 * normal_prob_below(x, mu, sigma)

def run_experiment():
    return [random.random() < 0.5 for _ in xrange(1000)]

def reject_fairness(experiment):
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1-p) / N)
    return p, sigma
def a_b_test_statistic(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    
    return (p_B - p_A) / math.sqrt(sigma_A**2 + sigma_B**2)

def B(alpha, beta):
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:
        return 0
    return x**(alpha - 1) * (1 - x)**(beta -1) / B(alpha, beta)