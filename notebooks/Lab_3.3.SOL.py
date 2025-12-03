import LP_LIB as lp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

global_seed = 0
n = 100
dim=1
lower=-1
upper=0
trueDeg = 3      #data generation (true) degree
degReg = 2     #regression degree. Closer to trueDeg => more accurate. 
w=None
#w=[1,1]
q=0
sigma=10
truth=True
np.random.seed(global_seed)


#generate and plot data
X, y, y_true = lp.polyDataGen(n,trueDeg,lower,upper,w,q,sigma,truth) 
#PLOT data + function - DON'T TOUCH
plt.scatter(X, y, color='blue', label='Noisy data', alpha=0.5)
plt.plot(X,y_true, color='green', label='True function', linewidth=2)
y_true = None #unset the true data, so there isn't any temptation :)


#do polynomial regression and plot it using np - sanity check
p = np.polyfit(X,y,degReg)                              #fit polynomial using coefficients
x_pol = np.linspace(-abs(2*lower),abs(2*upper),n*10)    #plot the poly regression in external points too, to see variance problems
yp = np.polyval(p,x_pol)                                #evaluate polynomial in plot points
control_loss = lp.polySquaredLoss(X,y,p)                #calculate the loss we're after, to check if gradient descent works
plt.plot(x_pol,yp,color='red', label='NumPy Polyfit', linewidth=2)
#plt.xlim([2*lower,2*upper])                            #xlim to plot prettier and understand
plt.ylim([4*min(y),4*max(y)])                           #ylim to plot prettier and understand


#SETUP FOR GRADIENT DESCENT POLYNOMIAL REGRESSION
#poly regression using GD
GD_weights = []
losses = []
w_0 = np.random.normal(0, 1, degReg + 1)  #gaussian initialization of weights
alpha=0.9    #learning rate
t_max=2000   
tol=1e-8
fixed_alpha = False 


# NORMALIZATION - #TODO try without normalization too
#X_max = np.max(np.abs(X))           #normalize X values to prevent numerical overflow  
#X_norm = X / X_max                  #normalize X to [-1, 1] range
#print(f"Normalized X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")

#p = np.polyfit(X, y, degReg)  #for comparison, fit numpy polynomial on normalized data too
#p_norm = np.polyfit(X_norm, y, degReg)  #for comparison, fit numpy polynomial on normalized data too
#control_loss_norm = lp.polySquaredLoss(X_norm, y, p_norm)
#print(f"Expected loss on normalized data: {control_loss_norm}")

# Better initialization: start with a good guess

#print(f"Starting gradient descent with initial weights: {w_0}")
#print(f"Target polynomial degree: {degReg}")



#execution of GD
GD_weights, losses = lp.gradientDescent(lp.polySquaredLossGradient,lp.polySquaredLoss,X,y,w_0,alpha,t_max,tol,fixed_alpha)


print('Expected loss:',control_loss)        #print np.poly loss
print(f"Final GD loss: {losses[-1]}")       #print final GD loss


#print(f"Final GD weights: {GD_weights[-1]}")
#print(f"NumPy polyfit loss (normalized): {control_loss_norm}")
#print(f"Difference in loss: {abs(losses[-1] - control_loss_norm):.6f}")


# Normalize x_pol for evaluation with GD model
#x_pol_norm = x_pol / X_max  # Use same normalization as training
#ypgd = np.polyval(GD_weights[-1], x_pol_norm)
ypgd = np.polyval(GD_weights[-1], x_pol)

plt.plot(x_pol, ypgd, color='yellow', linewidth=2)
plt.show()


plt.plot(range(len(losses)),losses)
#plot x axis line
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.yscale('log')
plt.xlabel('n. iterations')
plt.ylabel('loss')
plt.show()


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