import numpy as np
import numpy.linalg as LA



# just to remember data[:,2] = np.where( (data[:,0]*m+q-data[:,1]<0) ^ (np.random.rand(len(data[:,1]))<0.05), -1, 1 )

def linearRegression(x_train, y_train):
    """
    Implements linear regression using the closed-form solution.
    Parameters
    ----------
    x_train : array
        Training input data
    y_train : array
        Training output data
    
    Returns
        ----------
    w : array
        Estimated weights of the linear model
    """
    from numpy.linalg import inv
    #TODO SEPARATE IN FUNCTION TO ADD 1s as last column of X to learn b (intercept) as well

    w = inv(x_train.T @ x_train) @ x_train.T @ y_train
    return w

def ridgeRegression(x_train, y_train, lam):
    """
    Implements ridge regression using the closed-form solution.
    Parameters
    ----------
    x_train : array
        Training input data
    y_train : array
        Training output data
    lam : float
        Hyperparameter
    
    Returns
        ----------
    w : array
        Estimated weights of the linear model
    """
    import numpy as np
    import numpy.linalg as la
    from numpy.linalg import inv

    try:
        d = np.shape(x_train)[1]
    except IndexError:
        d = 1
    w = (inv(x_train.T@x_train+lam*np.eye(d))@x_train.T)@y_train
    return w

def squaredLoss(X,y,w):
    """
    Calcualtes squared loss for linear model with weights w on data X with targets y.
    Parameters
    ----------
    X : 2dimensional np.array
        Input data
    y : array
        Output data
    w : np.array
        Weights of the linear model
    Returns
    -------
    float
        Squared loss value 
    """
    return (np.linalg.norm((y-X@w),2)**2)/len(X)

def squaredLossGradient(X : np.ndarray, y : np.ndarray, w : np.array):
    """
    Calculates the gradient of the squared loss for linear model with weights w on data X with targets y.
    Parameters
    ----------
    X : np.array
        Input data
    y : np.array
        Output data
    w : array-like
        Weights of the linear model
    Returns
    -------
    np.array
        Gradient of the squared loss
    """
    #TODO fix overflow
    #print((X@w-y))
    return (2/len(X))*(X@w-y)@(X)

def polySquaredLoss(X, y, w):
    """
    Calculates squared loss for polynomial regression.
    Compatible with gradientDescent function interface.
    
    Parameters
    ----------
    X : array
        Input data (1D array of x values)
    y : array
        True output values
    w : array
        Polynomial coefficients [highest degree first, ..., constant term]
    
    Returns
    -------
    float
        Squared loss value
    """
    yp = np.polyval(w, X)  # calculate the polynomial (use the model on the data)
    loss = 0
    for i in range(len(X)):
        loss += (yp[i] - y[i])**2
        
    return loss/len(X)

def polySquaredLossGradient(X, y, w):
    """
    Calculates the gradient of the squared loss for polynomial regression.
    Compatible with gradientDescent function interface.
    
    For a polynomial model y = w[0]*x^(n-1) + w[1]*x^(n-2) + ... + w[n-1],
    this function computes the gradient of the mean squared error with respect 
    to the polynomial coefficients.
    
    Parameters
    ----------
    X : array
        Input data (1D array of x values)
    y : array
        True output values
    w : array
        Polynomial coefficients [highest degree first, ..., constant term]
    
    Returns
    -------
    array
        Gradient of the squared loss with respect to polynomial coefficients
    """
    n = len(X)              #dimension of input array
    deg = len(w) - 1        #maximum degree of regression polynomial
    y_pred = np.polyval(w, X)      #calculate polynomial predictions
    residuals = y_pred - y          #calculate residuals
    gradient = np.zeros_like(w)     #initialize gradient array, 1 element per polynomial coefficient
    
    # Build feature matrix for vectorized computation
    X_powers = np.column_stack([X**power for power in range(deg, -1, -1)]) #RIGA PAZZESCA COPILOT DRAGO
    # Vectorized gradient computation: gradient = (2/n) * X^T * residuals
    gradient = (2/n) * X_powers.T @ residuals
    return gradient

def gradientDescent(gradientFunction,lossFunction,X,y,w_0=None, alpha=0.1, t_max=1000, tol=1e-15, fixed_alpha=True):
    if w_0 is None:
        w_0 = np.ones(X.shape[1]) #if starting weights arent specified, generate 0s for every feature. #TODO fix
        
    w = w_0
    ws = []
    losses = []
    t0 = alpha
    for t in range(t_max):                  #stopper at t_max, maximum iteractions TODO fix
        if not fixed_alpha:
            alpha = t0/(t+1)                    #optional: update alpha each step, as per Cornell's Best Practices
        s = -alpha*gradientFunction(X,y,w)      #evaluate stepsize using learning rate (alpha) and the given gradient function
        w = w + s                               #update model weights
        ws.append(w.copy())                     #add to output (copy to avoid reference issues)
        #print('weights at iteration ',t)
        #print(w)
        losses.append(lossFunction(X,y,w))        #add to output
        #print(s)
        if(np.linalg.norm(s,2) < tol):      #if stepsize is smaller than tol, stop
            print("Converged in %d iterations at tol=%g" % (t, tol))
            return ws, losses                       
    print("Max iterations reached: %d" % t_max)
    return ws, losses

def adaGraD(gradientFunction,lossFunction,X,y,w_0=None, alpha=0.1, t_max=1000, tol=1e-15):
    if w_0 is None:
        w_0 = np.ones(X.shape[1]) #if starting weights arent specified, generate 0s for every feature. #TODO fix
    w = w_0
    ws = []
    regularization = np.zeros_like(w, dtype=np.float64)
    losses = []
    alpha = 1
    for t in range(t_max):                  #stopper at t_max, maximum iteractions TODO fix
        g = gradientFunction(X,y,w)      #evaluate stepsize using learning rate (alpha) and the given gradient function
        regularization += g*g
        s = -alpha*g/(np.sqrt(regularization + 0.1))
        w = w + s                               #update model weights
        ws.append(w.copy())                     #add to output (copy to avoid reference issues)
        #print('weights at iteration ',t)
        #print(w)
        losses.append(lossFunction(X,y,w))        #add to output
        #print(s)
        if(np.linalg.norm(s,2) < tol):      #if stepsize is smaller than tol, stop
            print("Converged in %d iterations at tol=%g" % (t, tol))
            return ws, losses                       
    print("Max iterations reached: %d" % t_max)
    return ws, losses

def secantMethod(loss, ws, datas, lables):
    """
    
    Secant method for optimization.
    This function approximates the next point in an linear trajectory
    using a secant-like approach in multi-dimensional space.
    
    Parameters
    ----------
    loss : callable
        Loss function that takes (data, labels, weights) and returns a scalar loss value.
    ws : list or array-like
        List of weight vectors used to compute the points in the space.
        The first element (ws[0]) is used as the reference point.
    datas : 2 dimensional np.array
        Input features or training data.
    lables : 1 dimensional np.array
        Target labels or ground truth values.
    
    Returns
    -------
    w_2 : array-like
        The computed secondary weight vector using the secant method approximation.
    Notes
    -----
    - The first weight vector ws[0] is treated as the initial reference point.
    - The direction vector is computed as the sum of differences of each points anche the initial.
    - The step size t_2 is determined by dividing the negative loss at the reference point
      by the cumulative direction vector.
    
    The theory behind:
    In iperspace the function of a iperline passing throw two points is T*(p1 - p2) + p1
    I add the loss of each point to create the points to work with,
    Than i create a line in the direction of the vector determinated by the sum
    of the difference from each point and the initial point. 
    Than intersect the line and the iperplane where lives the data without the loss by setting the loss to 0
    and compute the T value.
    Than evaluate the rimaning value that is the coordinates of the returning point
    """
    
    # prende in input un array di punti, assumo il primo come punto iniziale 
    direction_vector = sum([loss(datas, lables, ws[i]) - loss(datas, lables, ws[0]) for i in range(len(ws))])
    t_2 = -loss(datas, lables, ws[0])/direction_vector
    w_2 = t_2 * sum([ ws[i] - ws[0] for i in range(len(ws))]) + ws[0]
    return w_2

def GDSecantMethod(loss ,X ,y , ws, t_max, tol=1e-15):
    """
    Iteratively generate weight vectors using a secant-method update and collect them in a list.

    This function performs up to t_max iterations.
    Parameters
    ----------
    loss : callable
        A loss (or objective) function passed through to secantMethod. The exact expected
        signature is determined by the implementation of secantMethod, but it should accept
        the loss argument as the first parameter.
    X : 2 dimensional np.array
        Feature matrix / input data passed to secantMethod.
    y : 1 dimensional np.array
        Target vector / labels passed to secantMethod.
    ws : list-like
        Initial list (or sequence) of weight vectors. The function works on a shallow copy of
        this list, appending successive iterates to that copy; the original ws provided by the
        caller is not modified. The initial length of ws determines the sliding window size
        (n_dim) used when calling secantMethod.
    t_max : int
        Maximum number of iterations to perform.
    tol : float, optional (default=1e-15)
        Convergence tolerance. If the L2 norm (Euclidean norm) of a newly computed weight
        vector is less than tol, the function prints a convergence message and returns early.

    Returns
    -------
    ws : list
        A list containing the initial weight vectors (copied from the input ws) followed by
        the appended iterates produced during optimization. If convergence is reached before
        t_max iterations, the list returned contains all iterates up to and including the
        converged vector.

    Notes
    -----
    - The function relies on an external function secantMethod; its required call signature
      (in this code) is secantMethod(loss, ws_window, X, y), where ws_window is the most
      recent n_dim elements of the current ws copy.

    """
    ws = ws.copy()
    n_dim = len(ws)
    for i in range(t_max):
        w_next = secantMethod(loss, ws[-n_dim:], X, y)
        ws.append(w_next)
        if(np.linalg.norm(w_next,2) < tol):
            print("Converged in %d iterations" % i)
            return ws
    return ws
