import numpy as np

def sigmoid(x_i,w):
    #given the feature vector, the weights, calculate the sigmoid function of x_i and w.
    return 1/(1+np.exp(-np.dot(x_i,w))) #TODO +b

def logisticLoss(x,y,w):
    return sum(np.log( np.ones_like(y) + np.exp(-x @ w * y) ))
    # some of 1+e^{w*xi*yi}