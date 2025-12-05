#Lab 1: data generation and visualization for classification and regression.

#1.1: Generare dati sintetici (X,y) per problemi di regressione lineare utilizzando il modello y=wx+ϵ, dove ϵ è un rumore Gaussiano (funzione datagen).

#Prerequisites: numpy, matplotlib.pyplot

import numpy as np
import matplotlib.pyplot as plt
import LPEG.lpeg_data_generators as lp

"""
dim = 1 #number of features
n = 50  #number of data points
l = -10 #lower bound
u = 10 #upper bound
w = np.ones(dim) #weights (linear coefficients)
w = np.random.normal(0, 1, dim) #random weight (one coefficient per each feature)
#print(w)
X,y = lp.linear_data_generator(n,dim,l,u,w)  # Fixed: linDataGen -> linear_data_generator
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

np.random.seed(70)
#NEW TASK: create data, make it linearly separable.
n = 10 #points to be generated
N = 4 #classes to be generated
# Fixed: removed lower and upper parameters, they don't exist in gaussian_clouds_data_generator
d = lp.gaussian_clouds_data_generator(n, sparcity=10, n_classes=N, means=None, sigmas=None) #generate data
d[:,2] = np.mod(d[:,2],2)
print(d[:,2])

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



