import LP_LIB as lp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

global_seed = 0
n = 10
dim=1
lower=-1
upper=1
trueDeg = 2      #data generation (true) degree
degReg = 4     #regression degree. Closer to trueDeg => more accurate. 
w=None
#w=[1,1]
q=0
sigma=5
truth=True
np.random.seed(global_seed)
X, y, y_true = lp.polyDataGen(n,trueDeg,lower,upper,w,q,sigma,truth) #data generation. X is training data, y are generated values. y_true are the unable to obtain, true values used for plotting and mental sanity.

#PLOT data + function - DON'T TOUCH
plt.scatter(X, y, color='blue', label='Noisy data', alpha=0.5)
plt.plot(X,y_true, color='green', label='True function', linewidth=2)
y_true = None #unset the true data, so there isn't any temptation :)


#poly regression using np - sanity check
p = np.polyfit(X,y,degReg)                              #fit polynomial using coefficients
x_pol = np.linspace(-abs(2*lower),abs(2*upper),n*10)    #plot the poly regression in external points too, to see variance problems
#x_pol = X
yp = np.polyval(p,x_pol)                                #evaluate polynomial in plot points
control_loss = lp.polySquaredLoss(X,y,p)                #calculate the loss we're after, to check if gradient descent works
plt.plot(x_pol,yp,color='red', label='Polynomial Regression', linewidth=2)
#plt.xlim([2*lower,2*upper])                            #xlim to plot prettier and understand
plt.ylim([4*min(y),4*max(y)])                           #ylim to plot prettier and understand
print('Expected loss:',control_loss)                    #print the loss using np.poly*


#poly regression using GD
GD_weights = []
losses = []
#w_0=[1,1,1]   #initial weights
w_0=[0,0,0]
alpha=0.1
t_max=10000
tol=1e-15
GD_weights, losses = lp.gradientDescent(lp.polySquaredLossGradient,lp.polySquaredLoss,X,y,w_0,alpha,t_max,tol)


ypgd = np.polyval(GD_weights[-1],x_pol)

plt.plot(x_pol, ypgd, color='yellow', linewidth=2)
plt.show()


plt.plot(range(len(losses)),losses)
plt.xlabel('n. iterations')
plt.ylabel('loss')
plt.show()


#TODO NORMALIZE

"""
#plot the polynomial regression result
coeffs, losses = lp.gradientDescent(lp.polySquaredLossGradient,lp.polySquaredLoss,X,y,w_0=np.ones(1),alpha=0.01,t_max=10000)
gdyplo = np.polyval(coeffs,x_pol)
#set scale so IT IS FUCKING UNDERSTANDABLE
plt.plot(x_pol,gdyplo,color='orange', label='PolyReg GD', linewidth=2, scalex=1, scaley=1)
plt.title('Polynomial Regression using Numpy and Gradient Descent')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend()
plt.show()

"""

'''
#polynomial regression


plt.plot(range(len(losses)),losses,label='PolyReg Log-Loss',color='blue')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()



'''



'''
coeffs = np.polyfit(X,y,deg)
err = mean_squared_error(y_true,np.polyval(coeffs,X))
loss = lp.polySquaredLoss(X,y_true,coeffs)
print(err)
print(loss)
'''