#Nearest Neighbor Methods
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from functions import euclideanDistance,classificationPlot

def nearestNeighbor(x0,X,k):
    neighbor = np.zeros((k,n))
    dist = np.zeros((m,2))
    for i in range(m):
        dist[i] = np.array([i,euclideanDistance(x0,X[i])])
    index = dist[dist[:,1].argsort()][0:k,0]
    for i in range(k):
        neighbor[i] = X[int(index[i])]
    return neighbor

def getY(x0,X,Y):
    for i in range(Y.__len__()):
        if euclideanDistance(x0,X[i])==0:
            return Y[i]
    return -1

def model(x0,X,Y,k=5):
    sum = 0
    neighbor = nearestNeighbor(x0,X,k)
    for i in range(k):
        sum = getY(neighbor[i],X,Y)+sum
    return sum/k

features = pandas.read_csv('dataset.csv',usecols=['X','Z'])
labels = pandas.read_csv('dataset.csv',usecols=['LABEL'])
X = np.array(features)
Y = np.array(labels)
m,n = X.shape

redPoints = np.zeros((m,2))
greenPoints = np.zeros((m,2))
actualOne = np.zeros((m,2))
actualZero = np.zeros((m,2))
counter = 0

#Model parameter
k = 5

for i in range(m-1):
    if model(X[i],X,Y,k)>0.5:
        redPoints[i] = X[i]
        if Y[i]==0:
            actualZero[i] = X[i]
            counter += 1
        else:
            actualOne[i] = X[i]
    else:
        greenPoints[i] = X[i]
        if Y[i]==1:
            actualOne[i] = X[i]
            counter += 1
        else:
            actualZero[i] = X[i]

print("--Performance Evaluation--")
print("Neighbor dimension: ",k)
print("-----------------------")
print("Absolute mismatch: ",counter)
print("Relative mismatch: ",counter/m)

'''
#t = np.linspace(0,3.14,m)
#separationLine = np.sin(t)
fig,axs = plt.subplots(2)
axs[0].set_title("Model classification")
axs[0].scatter(redPoints[:,0],redPoints[:,1],color='red')
axs[0].scatter(greenPoints[:,0],greenPoints[:,1],color='green')
axs[1].set_title("Actual classification")
axs[1].scatter(actualZero[:,0],actualZero[:,1],marker='o')
axs[1].scatter(actualOne[:,0],actualOne[:,1],marker='x')
#plt.plot(t,separationLine)
plt.show()
'''

def classificationPlot(X,Y,k,model,res=50,progressBar=True):
    m,n = X.shape
    xx, yy = np.meshgrid(np.linspace(0, 3.14, res), np.linspace(0, 1, res))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
    x = np.zeros(n)
    pred = np.zeros((res,res))
    for i in range(res):
        if progressBar:
            print(i / res * 100, "%")
            for j in range(res):
                x[0] = xx[i, j]
                x[1] = yy[i, j]
                pred[i, j] = model(x,X, Y,k)
        else:
            for j in range(res):
                x[0] = xx[i, j]
                x[1] = yy[i, j]
                x[2] = 1
                pred[i, j] = model(x,X, Y, k)
    plt.figure()
    plt.pcolormesh(xx, yy, pred, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap=cmap_bold)
    plt.show()

classificationPlot(X,Y,1,model,res=100)