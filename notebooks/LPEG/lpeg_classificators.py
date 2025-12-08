import numpy as np
import numpy.linalg 

def sigmoid(x_i,w):
    #given the feature vector, the weights, calculate the sigmoid function of x_i and w.
    return 1/(1+np.exp(-np.dot(x_i,w))) #TODO +b


def safe_sigmoid(X,w): #numerically safe sigmoid function. Sets the output as eps if too low, 1-eps if too high.
    #TODO find out if this solves normalizing the dataset.
    eps = np.finfo(np.float64).eps
    h = 1/(1+np.exp(-np.dot(X,w))) #TODO +b
    h[h == 0] = eps
    h[h == 1] = 1-eps
    return h

def logistic_loss(X, y, w):
    h = safe_sigmoid(X, w)
    # Fix: Use negative log-likelihood and average over samples
    loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
    #print('loss: ', loss)
    return loss

def logistic_loss_gradient(X, y_true, w):
    h = safe_sigmoid(X, w)
    # Fix: Correct gradient formula
    gradient = np.mean(X.T * (h - y_true), axis=1)
    #print('X =', X)
    #print('Y = ', y_true) 
    #print('h =', h)
    #print('grad = ', gradient)
    return gradient
    

'''
import matplotlib.pyplot as plt

X = np.linspace(-10,10,100)
y = np.zeros(100)
for i in range(100):
    y[i] = sigmoid(X[i],1)

plt.plot(X,y)
plt.show()

'''