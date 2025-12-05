import numpy as np

def sigmoid(x_i,w):
    #given the feature vector, the weights, calculate the sigmoid function of x_i and w.
    return 1/(1+np.exp(-np.dot(x_i,w))) #TODO +b