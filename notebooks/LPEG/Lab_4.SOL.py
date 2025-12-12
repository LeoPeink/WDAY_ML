import numpy as np
import matplotlib.pyplot as plt
import lpeg_regressions as lpr
import lpeg_data_generators as lpg
import lpeg_preprocessing as lpp


n = 100
dim = 1
l = 0
u = 1
#w = [3,2]
w = None
w = [3]
q=8
sigma = 0.5
global_seed = 1
np.random.seed(global_seed)
#w_0 = [1,1]
w_0 = [1]
alpha = 0.1
t_max = 200
tol = 1e-6
lam = 0.5
fixed_alpha = False



X,y,y_true = lpg.linear_data_generator(n,dim,l,u,w,q,sigma,True) #generate linear data with noise

#add some outliars
num_outliers = 5
outlier_indices = np.random.choice(n, num_outliers, replace=False)
y[outlier_indices] += np.random.normal(0, 20*sigma, num_outliers)

#preprocessing step to add q to the data
#X = lpp.add_bias_term(X)

#preprocessing step to normalize the data
y = lpp.minmax_scaler(y,0,1)


'''
X0 = np.array(X[:,0])
X1 = np.array(X[:,1])
'''


ws,losses = lpr.gradientDescent(lpr.ridge_regression_gradient,lpr.ridge_regression_loss,X,y,lam,w_0,alpha,t_max,tol,fixed_alpha)
#ws,losses = lpr.adaGraD(lpr.ridge_regression_gradient,lpr.ridge_regression_loss,X,y,lam,w_0,alpha,t_max,tol)

w_pred = ws[-1]
y_pred = X@w_pred 



#plot original data
plt.figure(0)
plt.scatter(X[:,0],y, color='blue', label='Noisy data', alpha=0.5)
plt.plot(X[:,0],y_true, color='green', label='True function')
plt.plot(X[:,0],y_pred, color='red', label='Predicted function')
plt.legend()

#plot the loss over the iterations
plt.figure(1)
plt.plot(range(len(losses)),losses) #loss plot
plt.yscale('log')
plt.xlabel('n° iterations')
plt.ylabel('log(loss)')

#plot the weights over the iterations
plt.figure(2)
ws_array = np.array(ws)
for i in range(ws_array.shape[1]):
    plt.scatter(range(len(ws_array)),ws_array[:,i],label=f'w_{i}')
plt.xlabel('n° iterations')
plt.ylabel('weight value')
plt.legend()
plt.show()
