#LAB 2


#step 1: generate data from y=sin(pi*x)+eps
#   where eps ~ N(0,sigma)


import numpy as np
import LP_LIB as lp
import matplotlib.pyplot as plt

# Import the 'randint' function from the 'random' library for generating random integers
from random import randint

# Import the constant 'PI' from the 'math' library for mathematical calculations
from math import pi as PI

# Import mean_squared_error for evaluating regression models by calculating mean squared error
from sklearn.metrics import mean_squared_error

# Import train_test_split for splitting datasets into training and testing subsets
from sklearn.model_selection import train_test_split




#DATA PARAMETERS
global_seed = 42
np.random.seed(global_seed)  #random seed
n = 5000     #number of points       (observations)
dim = 1     #number of dimensions   (features)
l = 0      #lower bound for x
u = 2       #upper bound for x
sigma = 0.5 #noise variance
degs = 20 #numbers of polynomials to evaluate
X = np.linspace(l,u,n)  #creates x axis to be evaluated
eps = np.random.normal(0,sigma,n)   #creates the actual noise
y = (np.sin(np.pi*X)+eps)   #evaluates the function, generating data

#DATA OUTPUT
#plt.scatter(X,y)
#plt.show()

# splitting training and test data using sklearn
X_train, X_test, y_train, y_test = train_test_split(X,y ,random_state=global_seed,test_size=0.2)
#plot train,test sets on the same plot

plt.figure(0)


ax = plt.axes()
''''''
ax.scatter(X_train,y_train, label="Training set", c="red", alpha=0.3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(X_test,y_test, label="Test set", c="green", alpha=0.3)


MLE = []

#polynomial fitting of degree k - plot the polynomials on a domain bigger than the training set
#to show the behaviour outside the training set
for k in range(degs):
    cp = np.polyfit(X_train,y_train,k)
    yp = np.polyval(cp,X_train)
    xp = np.linspace(l,u,len(X_train))
    yp = np.poly1d(cp)(xp)
    ax.plot(xp,yp, label="k="+str(k))
    MLE.append(mean_squared_error(y_train,yp))

#print(p)
ax.legend()
plt.show()

plt.close(0)

plt.figure(1)
ax = plt.axes()
ax.plot(range(degs),MLE)
plt.show()


"""
X,y = lp.linDataGen(50,1,0,1,None,0.1)
print(X)
plt.scatter(X,y)
plt.show()
"""