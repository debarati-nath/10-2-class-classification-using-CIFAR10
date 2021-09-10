#Import libraries
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import os
import sys
from scipy.optimize import minimize
import mtrand
from scipy import stats
import scipy.optimize as opt

#Load dataset
cifar10_dir = 'C:/Users/Tuli/Desktop/cifar-10-batches-py'
def load_CIFAR_batch(filename):
datadict = pd.read_pickle(filename)
X = datadict['data']
Y = datadict['labels']
X=X.reshape(10000, 3072)
Y = np.array(Y)
return X, Y
def load_CIFAR10(ROOT):
xs = []
ys = []
f = os.path.join(ROOT, 'data_batch_1')
X, Y = load_CIFAR_batch(f)
xs.append(X)
ys.append(Y)
Xtr = np.concatenate(xs)
Ytr = np.concatenate(ys)
del X, Y
Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
return Xtr, Ytr, Xte, Yte

#Class
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Training and testing data
def get_CIFAR10_data(num_training = 10000, num_test = 10000, show_sample = True):
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train[y_train == 0] = 1
y_train[y_train != 0] = 0
return X_train, y_train, X_test, y_test
X_train_raw, y_train_raw, X_test_raw, y_test_raw = get_CIFAR10_data()
X_train_R = X_train_raw[0:10000, 0:1024]
X_train_B = X_train_raw[0:10000, 1024:2048]
X_train_G = X_train_raw[0:10000, 2048:]
X_train_raw=((0.2989*X_train_R)+(0.5870*X_train_B)+(0.1140*X_train_G))/255
X_test_R = X_test_raw[0:10000, 0:1024]
X_test_B = X_test_raw[0:10000, 1024:2048]
X_test_G = X_test_raw[0:10000, 2048:]
X_test_raw=((0.2989*X_test_R)+(0.5870*X_test_B)+(0.1140*X_test_G))/255

#Preprocessing
def preprocessing_CIFAR10_data(X_train, y_train, X_test, y_test):
mean_image = np.mean(X_train, axis = 0)
X_train -= mean_image
mean_image1 = np.mean(X_test, axis = 0)
X_test -= mean_image
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
return X_train, y_train, X_test, y_test
X_train, y_train, X_test, y_test = preprocessing_CIFAR10_data(X_train_raw, y_train_raw, X_test_raw, y_test_raw
print ('Train data shape : ', X_train.shape)
print ('Train labels shape : ', y_train.shape)
print ('Test data shape : ', X_test.shape)
print ('Test labels shape : ', y_test.shape)

#Sigmoid and cost function
def sigmoid(z):
g = 1 / (1 + np.exp(-z))
return g
def lr_cost_function(theta, X, y, l):
m, n = X.shape
mask = np.eye(len(theta))
# Skip the theta[0, 0] parameter when performing regularization
mask[0, 0] = 0
X_dot_theta = X.dot(theta)
J = 1.0 / m * (-y.T.dot(np.log(sigmoid(X_dot_theta))) - (1 - y).T.dot(np.log(1 - (sigmoid(X_dot_theta))))) + l / (2.0 * m) * np.sum(np.square(mask.dot(theta)))
grad = 1.0 / m * (sigmoid(X_dot_theta) - y).T.dot(X).T + 1.0 * l / m * (mask.dot(theta))
return J, grad

#Classification
def classification(X, y, num_labels, l):
m, n = X.shape
all_theta = np.zeros((num_labels, n + 1))
X = np.hstack((np.ones((m, 1)), X))
initial_theta = np.random.randn(n+1)
for i in range(0, 2):
num_labels = 0 if i == 0 else i
#options= {'maxiter': 1000}
result = opt.minimize(fun=lr_cost_function, x0=initial_theta, args=(X, (y==num_labels).astype(int), l),method='TNC', jac=True,options= {'maxiter': 5000})
all_theta[i,:] = result.x
return all_theta
num_labels= 2
l = 0.1
all_theta = classification(X_train_raw, y_train_raw, num_labels, l)

#Prediction
def predictOneVsAll(all_theta, X):
m = X.shape[0];
num_labels = all_theta.shape[0]
p = np.zeros(m)
X = np.concatenate([np.ones((m, 1)), X], axis=1)
p = np.argmax(sigmoid(X.dot(all_theta.T)), axis = 1)
return p
pred = predictOneVsAll(all_theta, X_train)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y_train) * 100))

def predict(all_theta, X):
m = X.shape[0];
num_labels = all_theta.shape[0]
p = np.zeros(m)
X = np.concatenate([np.ones((m, 1)), X], axis=1)
p = np.argmax(sigmoid(X.dot(all_theta.T)), axis = 1)
return p
y_test[y_test == 0] = 1
y_test[y_test != 0] = 0
pred = predict(all_theta, X_test)
print('Testing Set Accuracy: {:.2f}%'.format(np.mean(pred == y_test.ravel()) * 100))








