import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas
from functions import euclideanDistance,Id,norm,classificationPlot


def weights(X,x0,tau=0.1):
    m,n = X.shape
    w = np.zeros(m)
    for i in range(m):
        w[i] = euclideanDistance(X[i],x0)**2
    return np.exp(-w/(2*tau**2))

def logistic(x,theta):
    return 1/(1+np.exp(-np.dot(x,theta)))

def logLikelihood(input,theta,output):
    sum = 0
    for i in range(m):
        sum = output[i]*np.log(logistic(input[i],theta))+(1-output[i])*np.log(1-logistic(input[i],theta))+sum
    return sum

def gradient(theta,X,Y,point,l=0.1):
    m,n = X.shape
    w = np.zeros(m)
    for i in range(m):
        w[i] = weights(X,point)[i]
    W = np.diag(w)
    grad = X.T.dot(W @ (Y-logistic(X,theta)))-l*theta
    return grad

def hessian(theta,X,Y,point,l=0.0001):
    m,n = X.shape
    d = np.zeros(m)
    for i in range(m):
        d[i] = -weights(X,point)[i]*logistic(X[i],theta)*(1-logistic(X[i],theta))
    D = np.diag(d)
    hess = X.T @ D @ X-l*Id(3)
    return hess

# Load dataset
features = pandas.read_csv('dataset.csv',usecols=['X','Z'])
labels = pandas.read_csv('dataset.csv',usecols=['LABEL'])
features = np.array(features)
labels = np.array(labels)
m = features.__len__()

X = np.array([features.T[0,:],features.T[1,:],np.ones(m)]).T
Y = labels[:,0].reshape((m,1))
n = X.shape[1]

#Newton Method
bluePoints = np.zeros((m,2))
redPoints = np.zeros((m,2))

def train(X,Y,point):
    tolerance = 1
    g = np.ones(n) * np.inf
    theta = np.array([0, 0, 0]).reshape((3, 1))
    #print("Point: ",point[0:2])
    while norm(g) > tolerance:
        g = gradient(theta, X, Y, point=point)
        H = hessian(theta, X, Y, point=point)
        theta = theta - np.linalg.inv(H) @ g
        #print("Norm: ", norm(g))
    return np.round(logistic(point, theta))

def logit(X,Y,point):
    alpha = 0.01  # Learning rate
    epochs = 500
    theta = np.random.rand(3,1)
    for j in range(epochs):
        theta = theta + alpha * (X.T @ (Y - logistic(X, theta)))
        # print("l(theta) = ",logLikelihood(input=X,theta=theta,output=Y))
    return np.round(logistic(point,theta))


classificationPlot(X,Y,[train],res=50,progressBar=True)
