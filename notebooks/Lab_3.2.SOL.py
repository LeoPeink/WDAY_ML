import LPEG as lp
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib.patches import Patch


n = 100
dim = 2
l = 0
u = 1
w = [3,2]
#w = None
q=2
sigma = 0.5
global_seed = 1
np.random.seed(global_seed)

X,y,t_true = lp.linear_data_generator(n,dim,l,u,w,q,sigma,True) #generate linear data with noise

X0 = np.array(X[:,0])
X1 = np.array(X[:,1])


#WE LEARN B AS WELL - MOD X TO HAVE 1s
X = lp.add_bias_term(X)
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d').scatter(X0,X1,y)

#implement linear regression: find a hyperplane that best approximates the data.
w_lr = lp.linearRegression(X,y) #two weights

#w_0 = [-100,50]
w_0 = [0,0,0]
alpha = 0.1
t_max = 2000
tol = 1e-6

print(f"Linear Regression - Coefficients: {w_lr[:-1]}, Intercept: {w_lr[-1]}")

ws_lrgd, losses = lp.gradientDescent(lp.squaredLossGradient,lp.squaredLoss,X,y,w_0,alpha,t_max,tol,fixed_alpha=True)

print(f"Gradient Descent - Coefficients: {ws_lrgd[-1][:-1]}, Intercept: {ws_lrgd[-1][-1]}")

#plot losses
plt.figure()
plt.plot(range(len(losses)), losses, label='Squared Loss')
plt.yscale('log')
#plot the minimum loss line as dotted black line
plt.hlines(min(losses),0,len(losses), colors='black', linestyles='dashed', label=str('Minimum Loss (%.2e)' % min(losses)))
plt.axhline(0,0,len(losses), color='black')
plt.xlabel('Iterations')
plt.ylabel('Squared Loss')
plt.title('Squared Loss vs Iterations')
plt.legend()


#plot the weights and the targets
plt.figure()
ws_array = np.array(ws_lrgd)
plt.plot(range(len(ws_lrgd)), ws_array[:,0], color='r', label='Weight 0')
plt.plot(range(len(ws_lrgd)), ws_array[:,1], color='g', label='Weight 1')
plt.plot(range(len(ws_lrgd)), ws_array[:,2], color='b', label='Bias')
plt.hlines(w_lr,0,len(ws_lrgd), linestyles='dashed', colors=['r','g','b'], alpha=0.5, label='"true" (closed form LR) Weights')
plt.xlabel('Iterations')
plt.ylabel('Weights')
plt.legend()


'''
# Create meshgrid for surface plotting
x0_range = np.linspace(X0.min(), X0.max(), 20)
x1_range = np.linspace(X1.min(), X1.max(), 20)
X0_mesh, X1_mesh = np.meshgrid(x0_range, x1_range)

# Create feature matrix for the mesh (include bias term)
X_mesh = np.column_stack([X0_mesh.ravel(), X1_mesh.ravel(), np.ones(X0_mesh.size)])
# Predict outputs for the mesh
y_mesh = X_mesh @ w_lr
y_mesh2 = X_mesh @ ws_lrgd[-1]
# Reshape predictions back to meshgrid shape
y_mesh = y_mesh.reshape(X0_mesh.shape)
y_mesh2 = y_mesh2.reshape(X0_mesh.shape)
# Plot the surface
fig = plt.figure() #new figure
ax = fig.add_subplot(projection='3d') #3D axis
surf1 = ax.plot_surface(X0_mesh, X1_mesh, y_mesh, color='red', alpha=0.5)
surf2 = ax.plot_surface(X0_mesh, X1_mesh, y_mesh2, color='green', alpha=0.5)
ax.scatter(X0, X1, y, c='blue', alpha=0.5) #actual outputs
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('y')
# Add a simple legend using patches (surface collections don't map directly to legend labels)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='red', label='Linear Regression'),
				   Patch(color='green', label='GD final'),
				   Patch(color='blue', label='Data')])
plt.show()
'''



# Create meshgrid for surface plotting
x0_range = np.linspace(X0.min(), X0.max(), 2)
x1_range = np.linspace(X1.min(), X1.max(), 2)
X0_mesh, X1_mesh = np.meshgrid(x0_range, x1_range)

# Create feature matrix for the mesh (include bias term)
X_mesh = np.column_stack([X0_mesh.ravel(), X1_mesh.ravel(), np.ones(X0_mesh.size)])

# Configuration: Change this to show more or fewer intermediate steps
num_intermediate_steps = 5  # Change this number to show more/fewer steps

# Get weights at different stages
total_iterations = len(ws_lrgd)
# Create gradient from very light blue to deep blue
light_blue = [0.8, 0.9, 1.0, 1.0]  # Very light blue with alpha
deep_blue = [0.0, 0.0, 0.8, 1.0]   # Deep blue with alpha
colors = []
for i in range(num_intermediate_steps):
    ratio = i / (num_intermediate_steps - 1) if num_intermediate_steps > 1 else 0
    color = [light_blue[j] * (1 - ratio) + deep_blue[j] * ratio for j in range(4)]
    colors.append(color)

# Generate logarithmically spaced iterations including first and last
if num_intermediate_steps <= 1:
    iterations = [0, -1]
    stage_names = ['First', 'Final']
else:
    step_size = (total_iterations - 1) // (num_intermediate_steps - 1)
    iterations = [i * int(np.log(step_size)) for i in range(num_intermediate_steps - 1)] + [-1]
    stage_names = [f'Step {i+1}' for i in range(num_intermediate_steps - 1)] + ['Final']

# Create stages dictionary
stages = {}
for i, (iteration, name) in enumerate(zip(iterations, stage_names)):
    color = colors[i % len(colors)]  # Cycle through colors if we need more
    stages[name] = (iteration, color)

print(f"Total iterations: {total_iterations}")
print(f"Showing {num_intermediate_steps} steps at iterations: {[it if it >= 0 else total_iterations-1 for it in iterations]}")

# Plot the surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(projection='3d')

# Plot Linear Regression surface
y_mesh_lr = (X_mesh @ w_lr).reshape(X0_mesh.shape)
surf_lr = ax.plot_surface(X0_mesh, X1_mesh, y_mesh_lr, color='red', alpha=0.8, label='Linear Regression')
#make the surface thicker
# Plot GD surfaces at different stages
legend_handles = [Patch(color='red', label='Linear Regression')]

for stage_name, (iteration, color) in stages.items():
    weights = ws_lrgd[iteration]
    actual_iteration = iteration if iteration >= 0 else total_iterations - 1
    print(f"{stage_name} iteration ({actual_iteration}) weights: {weights}")
    
    y_mesh_stage = (X_mesh @ weights).reshape(X0_mesh.shape)
    ax.plot_surface(X0_mesh, X1_mesh, y_mesh_stage, color=color, alpha=0.5)
    legend_handles.append(Patch(color=color, label=f'GD {stage_name} (iter {actual_iteration})'))

ax.scatter(X0, X1, y, c='blue', alpha=0.75, s=20)
legend_handles.append(Patch(color='blue', label='Data'))

ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_zlabel('y')
ax.legend(handles=legend_handles)
ax.set_title(f'Gradient Descent Progress: {num_intermediate_steps} Steps')
plt.show()