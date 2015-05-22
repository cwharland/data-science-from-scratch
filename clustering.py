from __future__ import division
from lin_alg import *
import random

class KMeans:
    """k-means algo"""
    
    def __init__(self, k):
        self.k = k # number of clusters
        self.means = None # means of clusters
        
    def classify(self, input):
        """return the index of the cluster to closest to input"""
        return min(range(self.k),
                  key = lambda i: squared_distance(input, self.means[i]))
    
    def train(self, inputs):
        # choose k rand points as initials
        self.means = random.sample(inputs, self.k)
        assignments = None
        
        while True:
            # Find new assignments
            new_assignments = map(self.classify, inputs)
            
            # If nothing changed we're good to go
            if assignments == new_assignments:
                return
            
            # otherwise keep
            assignments = new_assignments
            
            # And compute new means based on assigments
            for i in range(self.k):
                # get points in cluster
                i_points = [p for p,a in zip(inputs, assignments) if a == i]
                
                # check for membership
                if i_points:
                    self.means[i] = vector_mean(i_points)


def squared_clustering_errors(inputs, k):
    """finds total square error for k"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = map(clusterer.classify, inputs)
    
    return sum(squared_distance(input, means[cluster])
              for input, cluster in zip(inputs, assignments))


def is_leaf(cluster):
    """a cluster is a leaf if it has len 1"""
    return len(cluster) == 1

def get_children(cluster):
    """returns children of the cluster if merged else exception"""
    if is_leaf(cluster):
        raise TypeError("a leaf cluster has no children")
    else:
        return cluster[1]
    
def get_values(cluster):
    """returns the value in the cluster (if leaf)
    or all values in leaf clusters below"""
    if is_leaf(cluster):
        return cluster
    else:
        return [value
                for child in get_children(cluster)
                for value in get_values(child)]
    
def cluster_distance(cluster1, cluster2, distance_agg = min):
    """compute all pairwise distances btw clusters
    and apply distance_agg to the list"""
    return distance_agg([distance(input1, input2)
                        for input1 in get_values(cluster1)
                        for input2 in get_values(cluster2)])

def get_merge_order(cluster):
    if is_leaf(cluster):
        return float('inf')
    else:
        return cluster[0]
    
def bottom_up_cluster(inputs, distance_agg = min):
    # we start with all leaf clusters (this is bottom up after all)
    clusters = [(input,) for input in inputs]
    
    # Don't stop until we have one cluster
    while len(clusters) > 1:
        # the two clusters we want to merge
        # are the clusters that are closest without touching
        c1, c2 = min([(cluster1, cluster2)
                     for i, cluster1 in enumerate(clusters)
                     for cluster2 in clusters[:i]],
                     key = lambda (x,y): cluster_distance(x, y, distance_agg))
        
        # the above is really inefficient in distance calc
        # we should instead "look up" the distance

        # once we merge them we remove them from the list
        clusters = [c for c in clusters if c != c1 and c != c2]

        # merge them with order = # of clusters left (so that last merge is "0")
        merged_cluster = (len(clusters), [c1, c2])

        # append the merge
        clusters.append(merged_cluster)
    
    return clusters[0]



def generate_clusters(base_cluster, num_clusters):
    clusters = [base_cluster]
    
    # keep going till we have the desired number of clusters
    while len(clusters) < num_clusters:
        # choose the last-merge
        next_cluster = min(clusters, key = get_merge_order)
        # remove it from the list
        clusters = [c for c in clusters if c != next_cluster]
        # add its children to the list (this is an unmerge)
        clusters.extend(get_children(next_cluster))
    
    return clusters