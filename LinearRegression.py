# Linear regression vs Polynomial Regression
from builtins import set

import matplotlib.pyplot as plt
import pandas
import numpy as np
from functions import norm,MSE,weights

np.set_printoptions(precision=3,suppress=True)
features = pandas.read_csv('regression.csv',usecols=['X'])
labels = pandas.read_csv('regression.csv',usecols=['Y'])
X = np.array(features)[:,0]
Y = np.array(labels)
m = X.__len__()

X = np.array([np.ones(m),X]).T
m,n = X.shape

#Batch Gradient Descent
def linearRegressionGD(X,Y):
    alpha = 0.01 #Learning rate
    epochs = 1000 #Tolerance rate
    theta = np.random.rand(n,1) #Parameters

    for i in range(epochs):
        y_hat = np.dot(X,theta)
        theta = theta - alpha * (1.0/m) * np.dot(X.T,y_hat-Y)
    return theta

#Locally weighted linear regression
def LWLinear(X,Y,tau):
    m,n = X.shape
    fit = np.zeros(m)
    w = np.zeros(m)
    theta = np.zeros(n)
    for i in range(m):
        #print(theta)
        for j in range(m):
            w[j] = weights(X,X[i],tau)[j]
        W = np.diag(w)
        theta = np.linalg.inv((X.T @ W @ X)) @ X.T @ W @ Y
        fit[i] = theta.T @ X[i]
    return fit

def polynomialFit(X,Y,k):
    Z = np.zeros((m,k+1))
    for i in range(m):
        for j in range(k+1):
            Z[i,j] = X[i,1]**j
    theta = np.linalg.inv(Z.T @ Z) @ Z.T @ Y
    fit = np.zeros(m)
    for i in range(m):
        fit[i] = theta.T @ Z[i]
    return fit

fit = polynomialFit(X,Y,k=1)
fit1 = polynomialFit(X,Y,k=4)
fit2 = polynomialFit(X,Y,k=8)
fit3 = LWLinear(X,Y,tau=0.1)

print("MSE Linear: ",MSE(fit,Y))
print("MSE 4th-degree: ",MSE(fit1,Y))
print("MSE 8th-degree: ",MSE(fit2,Y))
print("MSE LW: ",MSE(fit3,Y))

plt.scatter(X[:,1],Y)
plt.plot(X[:,1],fit,label='Linear')
#plt.plot(X[:,1],fit1)
plt.plot(X[:,1],fit2,label='8th-degree')
plt.plot(X[:,1],fit3,label='Lwlr')
plt.grid()
plt.legend(loc='lower left')
plt.show()

