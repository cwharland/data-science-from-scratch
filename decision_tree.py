from __future__ import division
from collections import Counter, defaultdict
import math
from functools import partial

def entropy(class_probabilities):
    """given list of class probs p_i, compute entropy"""
    return sum(-p * math.log(p, 2)
              for p in class_probabilities
              if p)

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
           for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)
    
# partition entropy is the weighted sum of each individual entropy
def partition_entropy(subsets):
    """computes the entropy across all subsets (list of lists)"""
    total_count = sum(len(subset) for subset in subsets)
    
    return sum(len(subset) * data_entropy(subset) / total_count
              for subset in subsets)

def partition_by(inputs, attribute):
    """input is pair (attr_dict, label)
    returns dict: attr_value -> inputs"""
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute] # key on value of given attribute
        groups[key].append(input) # add the input to this list
    return groups

def partition_entropy_by(inputs, attribute):
    """computes total entropy for a given attribute partitioning"""
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())

def classify(tree, input):
    """classify the input using the given decision tree"""
    
    # if it's a leaf node just spit out value
    if tree in [True, False]:
        return tree
    
    # otherwise this has an attr to split
    attribute, subtree_dict = tree
    
    subtree_key = input.get(attribute) # None if missing
    
    if subtree_key not in subtree_dict: # no tree for this key
        subtree_key = None
        
    subtree = subtree_dict[subtree_key] # get the right subtree
    
    return classify(subtree, input)


def build_tree_id3(inputs, split_candidates = None):
    
    if split_candidates is None: # first time through
        split_candidates = inputs[0][0].keys() # all keys are possible
        
    # Find all the True False
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues
    
    # Step 1) if pure then return label
    if num_trues == 0:
        return False
    if num_falses == 0:
        return True
    
    # Step 2) if we have no candidates left return majority vote
    if not split_candidates:
        return num_trues >= num_falses
    
    # Step 3) start recurse
    # find best split
    best_attribute = min(split_candidates,
                        key = partial(partition_entropy_by, inputs))
    
    # split on that
    partitions = partition_by(inputs, best_attribute)
    # remove this attr from the candidates
    new_candidates = [a for a in split_candidates
                     if a != best_attribute]
    
    # recurse
    subtrees = {attribute_value : build_tree_id3(subset, new_candidates)
               for attribute_value, subset in partitions.iteritems()}
    
    subtrees[None] = num_trues > num_falses # default case of missing value
    
    return (best_attribute, subtrees)