import numpy as np


def logisticLoss(x,y,w):
    return sum(np.log( np.ones_like(y) + np.exp(-x @ w * y) ))
    # some of 1+e^{w*xi*yi}