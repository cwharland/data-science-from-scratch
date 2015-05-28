from __future__ import division
from collections import defaultdict, Counter
import random

def fix_unicode(text):
    return text.replace(u"\u2019", "'")

def generate_using_bigrams():
    current = "."
    result = []
    while True:
        next_word_candidates = transitions[current] # all bigrams for current
        current = random.choice(next_word_candidates) # pick one
        result.append(current)
        if current == '.':
            return " ".join(result)


def generate_using_trigrams():
    current = random.choice(starts)
    prev = "."
    result = [current]
    
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next_word = random.choice(next_word_candidates)
        
        prev, current = current, next_word
        result.append(current)
        
        if current == ".":
            return " ".join(result)


def is_terminal(token):
    return token[0] != "_"

def expand(grammar, tokens):
    for i, token in enumerate(tokens):
        
        # skip over terminals
        if is_terminal(token):
            continue
            
        # if non-terminal choose replacement at random
        replacement = random.choice(grammar[token])
        
        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]
            
        return expand(grammar, tokens)
    
    return tokens

def generate_sentence(grammar):
    return expand(grammar, ["_S"])


def roll_a_die():
    return random.choice([1,2,3,4,5,6])

def direct_sample():
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2

def random_y_given_x(x):
    return x + roll_a_die()

def random_x_given_y(y):
    if y <= 7:
        return random.randrange(1,y)
    else:
        return random.randrange(y - 6, 7)
    
def gibbs_sample(num_iters = 100):
    x, y = 1, 2 # arbitrary
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y

def compare_distributions(num_samples = 1000):
    counts = defaultdict(lambda: [0,0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[gibbs_sample()][1] += 1
    return counts


# works by working backwards through the cumulative dist
def sample_from(weights):
    """returns i with probability weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total*random.random() # uniform between 0 and total
    for i, w in enumerate(weights):
        rnd -= w # find the smallest i
        if rnd <= 0: # s.t. weights[0] + ... + weights[i] >= rnd
            return i
        
def p_topic_given_document(topic, d, alpha = 0.1):
    """fraction of words in document _d_
    that are assigned to _topic_ (plus some bit)"""
    
    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K*alpha))

def p_word_given_topic(word, topic, beta = 0.1):
    """fraction of words assigned to _topic_
    that equal _word_"""
    
    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W*beta))

def topic_weight(d, word, k):
    """given a document and a word in that doc,
    return the weight for the kth topic"""
    
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)

def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k)
                        for k in range(K)])