import numpy as np


def rescale(x,new_min,new_max,old_min=0,old_max=1):
    """
    Rescales x from [old_min, old_max] to [new_min, new_max]
    
    Parameters
    ----------
    x : float
        Value to be rescaled
    new_min : float
        New minimum value of the rescaled x
    new_max : float
        New maximum value of the rescaled x
    old_min : float
        Old minimum value of x (default 0)
    old_max : float
        Old maximum value of x (default 1)
        
    Returns
    -------
    float
        Rescaled value of x
    """
    #TODO check for edge cases and 0 division
    if old_max == old_min:
        raise ValueError("old_max and old_min cannot be the same")
    for i in range(len(x)):
        return (new_max - new_min)*(x[i] - old_min)/(old_max-old_min) + new_min
    
def linDataGen(n,dim=1,lower=0,upper=1,w=None,sigma=0):
    """
    Generates either clean or noisy, linearly-generated data.
    Formally returns y,X where y=wX + eps, where eps is gaussian noise.

    Parameters
    ----------
    n : int
        Number of points to be generated (num. of rows, entries)
    dim : int
        Dimension of each data sample (num. of columns, features)
    lower : float
        Lower bound for the domain of the data points
    upper : float
        Upper bound for the domain of the data points
    w : float array of dim dim
        Vector of weights of the linear model
    sigma : float
        Standard deviation of the noise eps
    
    Returns
    ----------
    X : array
        Generated input data
    y : array
        Generated output data
    """
    if w is None:
        w = np.ones(dim)
    X = np.zeros((n,dim)) #creates "n" points, empty
    for i in range(n): #for each point
        X[i,:] = (np.random.uniform(lower,upper,dim))
    #X = np.random.rand(n,dim)
    eps = np.random.normal(0,sigma,n)
    print(eps)
    y = np.dot(X,w)+eps
    return X,y
"""
#DEMO linDataGen
import numpy as np
import Z_LIB as zl
import matplotlib.pyplot as plt
dim = 1 #number of features
n = 50  #number of data points
l = 0 #lower bound
u = 1 #upper bound
w = np.ones(dim) #weights (linear coefficients)
#w = np.random.normal(0, 1, dim) #random weight (one coefficient per each feature)
#print(w)
X,y = linDataGen()
print(X)
print("w:")
print(w)
plt.scatter(X,y)
plt.show()
"""

def gaussianDataGen(n,sparcity=1,classes=2,flip=0,labels=None,means=None,sigmas=None): #todo sistema sigmas
    """
    Generate N Gaussian clouds of 2D points.

    Each cloud contains `n` points distributed according to a multivariate normal
    with mean vector `means` and covariance matrix `sigmas`.  
    Optionally, a percentage of labels can be randomly flipped.

    Parameters
    ----------
    n : int
        Number of points per cloud.
    sparcity : float, optional
        Average distance between the centers of the clouds. Default is 1.
    classes : int, optional
        Number of Gaussian clouds (and corresponding labels). Default is 2.
    flip : float, optional
        Percentage (0–100) of points whose labels will be randomly flipped. Default is 0.
    labels : list of int, optional
        List of labels to assign to each cloud.  
        If None, defaults to `[-1, 1]` for classes=2, or `range(classes)` otherwise.
    means : list of float, optional
        List of mean coordinates for each cloud (length 2N).  
        If None, random centers are generated within the range `[0, sparcity]`.
    sigmas : array-like of shape (classes), optional
        Will be used to build the covariance matrix (or list of matrices) for each cloud. Each element
        represents the covariance inside individual clouds.  
        If None, uses the 2×2 identity matrix `np.eye(2)`.

    Returns
    -------
    d : ndarray of shape (classes * n, 3)
        Array containing generated points and labels.  
        Columns are `[x, y, label]`.

    Examples
    --------
    >>> d = gCloudDataGen(n=100, classes=3, sparcity=5, flip=10)
    >>> d.shape
    (300, 3)

    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(d[:,0], d[:,1], c=d[:,2])
    >>> plt.show()
    """
    
    if means is None:
        #means = np.arange(2*N)
        means = np.random.randint(0,sparcity,2*classes)
    if sigmas is None: #TODO sistema
        sigmas=np.eye((2))
    if labels is None:
        labels = np.arange(classes)
        if classes ==2:
            labels = [-1,1]
    
    d = np.zeros((classes*n, 3))  # x, y, label

    for i in range(classes):  #for each cloud (= label)
        cov = (np.eye(2)*np.square(sigmas[i]))
        points = np.random.multivariate_normal([means[i],means[i+1]],cov,n)
        d[i*n : n*(i+1) , : ] = np.hstack([points, np.full((n,1),labels[i])])
        
    for i in range(classes*n):
        monetina = np.random.randint(0,100) #lancia monetina (numero da 0 a 100)
        if monetina < flip:                 #se esce testa (cioe' se il numero e' minore della percentuale di flip)     
            d[i][2] = labels[(monetina)%classes]  #cambia etichetta
    return d
'''
#DEMO gCloudDataGen
import Z_LIB.py as zl
d = zl.gCloudDataGen(n=50, classes=2, sparcity=10, flip=0, sigmas=[1,3], labels=[-1,1], means=[0,0,5,5])
#print(d)
# plotting the generated dataset
fig, ax = plt.subplots()
ax.scatter(d[:,0], d[:,1], c=d[:,2])
ax.set_title('Data')
plt.show()
'''