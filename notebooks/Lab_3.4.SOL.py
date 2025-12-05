import LPEG as lp
import numpy as np
import matplotlib.pyplot as plt


#plotting of sigmoid function

n = 100
lower = -10
upper = 10
w = [1]
y = np.zeros(n)
X = np.linspace(lower,upper,n)
for i in range(n):
    y[i] = lp.sigmoid(X[i],w)
    

    
plt.plot(X,y)
plt.show()