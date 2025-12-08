import LPEG as lp
import numpy as np
import matplotlib.pyplot as plt


    


#01loss - misclassification error / loss
def zero_one_loss(y_pred : np.ndarray , y_true : np.ndarray , sum:bool = True): 
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    loss = (y_pred != y_true).astype(int)
    if sum:
        return np.sum(loss)
    return loss





np.random.seed(0)
n = 300
sparcity = 5
n_clouds = 2
flip = 0
labels = [0,1]
means = None
sigmas = None

d = lp.gaussian_clouds_data_generator(n,sparcity,n_clouds,flip,labels,means,sigmas)

X0 = d[:,0]
X1 = d[:,1]
X = d[:,:-1]
y = d[:,2]
w_0 = [1,1]


print(d)
'''
#DEBUG sigmoid plot
xs = np.linspace(-1,1,100)
ys = lp.sigmoid(xs,10)
plt.plot(xs,ys)
plt.show()
'''


'''
#DEBUG dataset print
print('X0 =',X0)
print('X1 =',X1)
print('label =',y)

print('X shape =',np.shape(X))
print('y shape =',np.shape(y))
print('w shape =',np.shape(w))
'''

alpha = 0.1
t_max = 200
tol = 1e-6
fixed_alpha = True

print(y)

ws, losses = lp.adaGraD(lp.logistic_loss_gradient,lp.logistic_loss,X,y,w_0,alpha,t_max,tol)




fig, ax = plt.subplots()
# ensure the scatter uses the full [0,1] range so single-point probabilities map to the same colormap
sc = ax.scatter(X0,X1,c=y,cmap='bwr',alpha=0.5, vmin=0, vmax=1)

# Get the final weights from gradient descent
w = ws[-1]
#ONCLICK EVENT PLOT
def onclick(event):
    if event.inaxes != ax:  # Check if click is within the plot area
        return
    X_click = np.array([event.xdata, event.ydata])
    #y_t = float(np.squeeze(lp.sigmoid(X_click, w)))
    y_t = lp.sigmoid(X_click,w)
    #print coordinates, predicted probability, color (label) and prediction certainty
    #prediction certainty:
    certainty = y_t if y_t > 0.5 else 1 - y_t
    print('Clicked at (%.2f, %.2f) - Predicted class: %s, probability: %.3f' % (event.xdata, event.ydata,('blue' if y_t <= 0.5 else 'red'), y_t))
    print('Prediction certainty: %.3f' % (certainty*100) + '%')
    ax.scatter(event.xdata, event.ydata, c=[y_t], marker='o', alpha=0.5, cmap='bwr', vmin=0, vmax=1, edgecolor='k')
    plt.draw()  # Refresh the plot to show the new point

plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Generated Dataset - Click to Classify Points')
#label for classes
plt.colorbar(sc, ticks=[0,1], label='Class Label')
#NB: cmap bwr means blue for 0, red for 1

cid = fig.canvas.mpl_connect('button_press_event', onclick)

x_decision_boundary = np.linspace(min(X0), max(X0), 100)
y_decision_boundary = -(w[0] * x_decision_boundary) / w[1]
plt.plot(x_decision_boundary, y_decision_boundary, color='green', linestyle='--', label='Decision Boundary')
plt.legend()

plt.show()




plt.figure(1)
plt.plot(range(len(losses)),losses)
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('Logistic Loss')
plt.title('Logistic Loss vs Iterations')
plt.axhline(min(losses), color='black', linestyle='dashed', label='Minimum Loss (%.2e)' % min(losses))
plt.legend()


plt.figure(2)
#plot the weights over iterations
ws_array = np.array(ws)
plt.plot(range(len(ws)), ws_array[:,0], color='r', label='Weight 0')
plt.plot(range(len(ws)), ws_array[:,1], color='g', label='Weight 1')
plt.xlabel('Iterations')
plt.ylabel('weight value')
plt.legend()
# remove log scale for weights (uncomment if log scale is desired)
# plt.yscale('log')





plt.show()

''' 


loss = zero_one_loss(y,y_t,True)

print(loss)
w = [1]





loss = logistic_loss(X,w,y)
print()



plt.plot(X,y)
'''