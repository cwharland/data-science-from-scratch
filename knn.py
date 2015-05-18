from __future__ import division
from collections import Counter
import random
from lin_alg import *
from stats import *

def raw_majority_vote(labels):
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def majority_vote(labels):
    """assumes labels sorted by distance ASC"""
    vote_counts = Counter(labels)
    
    winner, winner_count = vote_counts.most_common(1)[0]
    
    num_winners = len([count
                      for count in vote_counts.values()
                      if count == winner_count])
    
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1]) # remove the furthest label and repeat
    
# now easy to build knn
def knn_classify(k, labeled_points, new_point):
    """labeled points are (point, label)"""
    
    # sort by distance ASC
    by_distance = sorted(labeled_points,
                        key = lambda (point, _): distance(point, new_point))
    
    # make the nearest neighbor label list
    k_nearest_lables = [label for _,label in by_distance[:k]]
    
    # vote
    return majority_vote(k_nearest_lables)

def random_point(dim):
    return [random.random() for _ in range(dim)]

def random_distances(dim, num_pairs):
    return [distance(random_point(dim), random_point(dim))
           for _ in range(num_pairs)]