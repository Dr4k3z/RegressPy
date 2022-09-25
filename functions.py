import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def norm(x):
    return np.sqrt(np.dot(x.T,x))

def MSE(actual,model):
    m = actual.__len__()
    sum = 0
    for i in range(m):
        sum = sum + (model[i]-actual[i])**2
    return sum/2

def euclideanDistance(x,y):
    return norm(x-y)

def Id(n):
    id = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                id[i] = 1
            else:
                id[i] = 0
    return id

def weights(X,x0,tau=0.1):
    m,n = X.shape
    w = np.zeros(m)
    for i in range(m):
        w[i] = euclideanDistance(X[i],x0)**2
    return np.exp(-w/(2*tau**2))

def classificationPlot(X,Y,listModels,res=50,progressBar=True):
    m,n = X.shape
    xx, yy = np.meshgrid(np.linspace(0, 3.14, res), np.linspace(0, 1, res))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    x = np.zeros(n)
    pred = np.zeros((res,res))
    for model in listModels:
        for i in range(res):
            if progressBar:
                print(i / res * 100, "%")
                for j in range(res):
                    x[0] = xx[i, j]
                    x[1] = yy[i, j]
                    x[2] = 1
                    pred[i, j] = model(X, Y, x)
            else:
                for j in range(res):
                    x[0] = xx[i, j]
                    x[1] = yy[i, j]
                    x[2] = 1
                    pred[i, j] = model(X, Y, x)
        plt.figure()
        plt.pcolormesh(xx, yy, pred, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap=cmap_bold)
    plt.show()
