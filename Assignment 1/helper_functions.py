import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_der(x):
  return sigmoid(x) * (1-sigmoid(x))

def relu(x):
  return np.maximum(0, x)

def relu_der(x):
  x[x>0] = 1
  x[x<=0] = 0
  return x

def tanh(x):
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_der(x):
  return 1 - tanh(x) ** 2

def identity(x):
  return x

def identity_der(x):
  return np.ones(x.shape)

def get_activation(act):
  if act == 'sigmoid':
    act = sigmoid
    act_der = sigmoid_der
  elif act == 'ReLU':
    act = relu
    act_der = relu_der
  elif act== 'tanh':
    act = tanh
    act_der = tanh_der
  elif act == 'identity':
    act = identity
    act_der = identity_der
  else:
    raise Exception('Incorrect Activation Function')
  return act, act_der

def get_loss_func(loss_func):
  if loss_func == 'cross_entropy':
    loss_func_ = cross_entropy
    loss_func_der = cross_entropy_der
  elif loss_func == 'mean_squared_error':
    loss_func_ = mean_squared_error
    loss_func_der = mean_squared_error_der
  else:
    raise Exception('Incorrect Loss Function')
  return loss_func_, loss_func_der

def softmax(x):
  xsum = np.sum(np.exp(x), axis=1, keepdims=True)
  res = np.exp(x)/xsum
  return res

def cross_entropy(y, yhat):
  epsilon = 1e-30
  losses = -np.sum(y * np.log(yhat + epsilon), axis=1)
  return np.mean(losses)

def cross_entropy_der(y, yhat):
  return -(y - yhat)

def mean_squared_error(y, yhat):
  return np.mean(np.sum((y-yhat)**2, axis=1))

def mean_squared_error_der(y, yhat):
  s = np.sum(((y-yhat) * yhat), axis=1, keepdims=True)
  return yhat * (s - (y - yhat))

def accuracy(y, yhat):
  return np.sum(np.argmax(y, axis=1) ==  np.argmax(yhat, axis=1)) / len(y)

def get_dataset(dataset):
    if dataset == 'fashion_mnist':
        from keras.datasets import fashion_mnist
        (xfull, yfull), (xtest, ytest) = fashion_mnist.load_data()
        class_labels = {0: 'T-Shirt',
                        1: 'Trouser',
                        2: 'Pullover',
                        3: 'Dress',
                        4: 'Coat',
                        5: 'Sandal',
                        6: 'Shirt',
                        7: 'Sneaker',
                        8: 'Bag',
                        9: 'Ankle Boot'}

    elif dataset == 'mnist':
        from keras.datasets import mnist
        (xfull, yfull), (xtest, ytest) = mnist.load_data()
        class_labels = {0: '0',
                        1: '1',
                        2: '2',
                        3: '3',
                        4: '4',
                        5: '5',
                        6: '6',
                        7: '7',
                        8: '8',
                        9: '9'}

    else:
        raise Exception('Incorrect Dataset Name')

    # xtrain, xval, ytrain, yval = train_test_split(xfull, yfull, test_size = 0.1, random_state=2)
    np.random.seed(seed=2)
    l = xfull.shape[0]
    test_size = 0.1
    ltrain = int(l * test_size)
    idxs = np.arange(l)
    np.random.shuffle(idxs)
    xval= xfull[idxs[:ltrain]]
    yval = yfull[idxs[:ltrain]]
    xtrain = xfull[idxs[ltrain:]]
    ytrain = yfull[idxs[ltrain:]]
    return (xtrain, ytrain), (xval, yval), (xtest, ytest), class_labels

def scale_dataset(xtrain, ytrain, xval, yval, xtest, ytest, scaling):
    if scaling == 'min_max':
      xtrain_inp = xtrain.reshape((xtrain.shape[0], -1))/255.0
      xval_inp = xval.reshape((xval.shape[0], -1))/255.0
      xtest_inp = xtest.reshape((xtest.shape[0], -1))/255.0
      ytrain_inp = np.array(pd.get_dummies(ytrain))
      yval_inp = np.array(pd.get_dummies(yval))
      ytest_inp = np.array(pd.get_dummies(ytest))

    elif scaling == 'standard':
      xtrain_inp = xtrain.reshape((xtrain.shape[0], -1))
      xval_inp = xval.reshape((xval.shape[0], -1))  
      xtest_inp = xtest.reshape((xtest.shape[0], -1))     
      mu = xtrain_inp.mean(axis=0)
      sigma = xtrain_inp.std(axis=0)
      xtrain_inp = (xtrain_inp - mu) / sigma
      xval_inp = (xval_inp - mu) / sigma
      xtest_inp = (xtest_inp - mu) / sigma
      ytrain_inp = np.array(pd.get_dummies(ytrain))
      yval_inp = np.array(pd.get_dummies(yval))
      ytest_inp = np.array(pd.get_dummies(ytest))

    else:
      raise Exception('Incorrect Data Scaling Input')

    return (xtrain_inp, ytrain_inp), (xval_inp, yval_inp), (xtest_inp, ytest_inp)