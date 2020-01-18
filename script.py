import pandas as pd
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

digits_data=load_digits()
print(digits_data.keys())
img=digits_data['images']
labels=digits_data['target']
labels=labels.reshape(1,labels.shape[0])

img_flatten=img.reshape(img.shape[0],-1).T
img_std=img_flatten/255

train_x=img_std[:,:1400]
train_y=labels[:,:1400]

test_x=img_std[:,1400:]
test_y=labels[:,1400:]


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s=1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w=np.zeros((dim,1))
    b=0
    
    return w,b

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation.

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px, number of examples)
    Y -- true "label" vector (containing digits ) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    """
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    cost = (1/m)*(np.sum(-(Y*np.log(A))-((1-Y)*np.log(1-A)))) 
    dz= (A-Y)
    dw = (1/m)*(np.dot(X,dz.T))
    db = (1/m)*(np.sum(dz))
    grads={ 'dw':dw,'db':db}
    
    return grads,cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px, number of examples)
    Y -- true "label" vector (containing digits), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    """
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w-(learning_rate*dw)
        b = b-(learning_rate*db)
        if i % 10 == 0:
            costs.append(cost)
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0][i]<0.5:
            Y_prediction[0][i]=0
        elif A[0][i]>0.5:
            Y_prediction[0][i]=1  
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 100, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the functions implemented.
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train))))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)))) 
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d

if __name__=='__main__':
    d = model(train_x, train_y, test_x, test_y, num_iterations = 100, learning_rate = 0.5, print_cost = True)
   

