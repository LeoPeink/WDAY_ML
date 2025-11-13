#Lab 1: data generation and visualization for classification and regression.

#1.1: Generare dati sintetici (X,y) per problemi di regressione lineare utilizzando il modello y=wx+ϵ, dove ϵ è un rumore Gaussiano (funzione datagen).

#Prerequisites: numpy, matplotlib.pyplot

import numpy as np
import matplotlib.pyplot as plt
import notebooks.LP_LIB as lp


    
def datagen(d,points,lower,upper,weight,sigma):
    """
    Returns y,X where y = wX + eps, where eps is gaussian noise.
    Parameters:
    ----------
    dim : int
        Dimension of each data sample (num. of columns, features)
    points : int
        Number of points to be generated (num. of rows, entries)
    lower : float
        Lower bound for the domain of the data points
    upper : float
        Upper bound for the domain of the data points
    weight : float array of dim d
        Vector of weights of the linear model
    sigma : float
        Standard deviation of the noise eps
    """
    X = np.zeros((points,d)) #create empty array 
    for i in range(points):
        X[:,i] = np.random.uniform(lower,upper,d)
    eps = np.random.normal(0,sigma,points)
    return 0


def linDataGen(dim,n,lower,upper,w,sigma=1):
    """
    Returns y,X where y=wX. Clean linear data.
    """
    X = np.zeros((n,dim)) #creates "n" points, empty
    for i in range(n): #for each point
        X[i,:] = (np.random.uniform(lower,upper,dim))
    #X = np.random.rand(n,dim)
    eps = np.random.normal(0,sigma,n)
    print(eps)
    y = np.dot(X,w)+eps
    return X,y

#def gCloudDataGen(ns,N=2,means=[0,1],sigmas=np.full((2,2),1)): #todo sistema sigmas



"""
dim = 1 #number of features
n = 50  #number of data points
l = -10 #lower bound
u = 10 #upper bound
w = np.ones(dim) #weights (linear coefficients)
w = np.random.normal(0, 1, dim) #random weight (one coefficient per each feature)
#print(w)
X,y = linDataGen(dim,n,l,u,w)
print(X)
print("w:")
print(w)

# plotting the generated dataset
fig, ax = plt.subplots()
ax.scatter(X, y, c='b')
ax.set_title('Data')
plt.ylim([l, u])
plt.show()
"""



"""

dim = 1 #number of features
n = 50  #number of data points
l = -10 #lower bound
u = 10 #upper bound
w = np.ones(dim) #weights (linear coefficients)
w = np.random.normal(0, 1, dim) #random weight (one coefficient per each feature)
#print(w)


"""
np.random.seed(70)
#NEW TASK: create data, make it linearly separable.
n = 10 #points to be generated
N = 4 #classes to be generated
d = lp.gaussianDataGen(n,1,N,5,sigmas=[1,3,7,0.01],labels=[1,1,0,0]) #creates two clouds with the same label
d[:,2] = np.mod(d[:,2],2)
print(d[:,2])
#RANDOM SEED


#print(d)
# plotting the generated dataset
fig, ax = plt.subplots()
ax.scatter(d[:,0], d[:,1], c=d[:,2])
ax.set_title('Data')
plt.show()



m = 1
q = 0

"""
x = d[:,0]
y = d[:,1]

for i in range(N*n):
    #print('x =' +str(d[i][0]) + 'y =' +str(d[i][1]) )
    #print(d[i])
    x=d[i][0]
    y=d[i][1]
    if m*x+q-y >= 0: #
        d[i][2] = -1
    else:
         d[i][2] = 1
   """      
 
#print(d[:][0]) #tutti gli elementi della prima riga
#print(d[:,0]) #tutte i values della prima colonna (tutte le x)




"""


#default plot with same scale on x and y axis
#subplot y=mx+q
#plt.axline((0,q), slope=m, color='r', linestyle='--')
#plt.axline((0,q),(1,1), color='r', linestyle='--') #y=x
#plot exponential curve

#assign data linear, given data, m and q    
d[:, 2] = np.where(m * d[:, 0] + q - d[:, 1] > 0, -1, 1)

#subplot y=mx+q
fig, axs = plt.subplots(3)
fig.suptitle('Various separations')
axs[0].scatter(d[:,0],d[:,1], c=d[:,2]) #first cloud, linearly separable
axs[0].axline((0,q), slope=m, color='r', linestyle='--')
axs[0].set_title('Linear separation')



#assign data exponential, given e (exponent)
e = 2
d[:, 2] = np.where(d[:, 1] - np.exp(1)**d[:,0] < 0, -1, 1)

#subplot y=e^x
axs[1].set_title('Exponential separation')
#axs[1].axis('equal')
axs[1].scatter(d[:,0],d[:,1], c=d[:,2]) #second cloud, exponentially separable
x_vals = np.linspace(min(d[:,0]), max(d[:,0]), 100)
y_vals = np.exp(1)**x_vals
axs[1].plot(x_vals, y_vals, color='g')

#assign data sinusoidal
x = d[:, 0]
y = d[:, 1]
#d[:, 2] = np.where(d[:, 1] - np.exp(1)**d[:,0] < 0, -1, 1)
d[:,2] = np.where(y-np.sin(x)<0,-1,1)
x_vals = np.linspace(min(d[:,0]), max(d[:,0]), 100)
y_vals = np.sin(x_vals)
axs[2].scatter(d[:,0],d[:,1], c=d[:,2]) #first cloud, linearly separable
axs[2].plot(x_vals, y_vals, color='g')

"""
plt.show()