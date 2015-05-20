from __future__ import division
import math
import random
from lin_alg import dot


def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    """returns 1 if perceptron fires, 0 if not"""
    calculation = dot(weights, x) + bias
    return step_function(calculation)


def sigmoid(t):
    return 1.0 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(dot(weights, inputs))

# NN is then a list of lists of lists (where layers have neurons that have weights)

def feed_forward(neural_network, input_vector):
    """NN (list of lists of lists)
    returns output from forward prop"""
    
    outputs = []
    
    # for each layer add bias, compute output, pass on
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                 for neuron in layer]
        outputs.append(output)
        
        input_vector = output
        
    return outputs

def backpropogate(network, input_vector, targets):
    
    hidden_outputs, outputs = feed_forward(network, input_vector)
    
    # recall derivative of sigmoid is same as logit
    output_deltas = [output * (1 - output) * (output - target)
                    for output, target in zip(outputs, targets)]
    
    # adjust weights gradient descent style
    for i, output_neuron in enumerate(network[-1]):
        # iterate over hidden layers
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output
            
    # now propogate this change backward
    hidden_deltas = [hidden_output * (1 - hidden_output) * 
                    dot(output_deltas, [n[i] for n in output_layer])
                    for i, hidden_output in enumerate(hidden_outputs)]
    
    # adjust weights
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

def predict(input):
    return feed_forward(network, input)[-1]

    