import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Calculationg Weight Matrix Using e^(-(x-x0)^2/2*r^2)
def kernel(point,xmat, k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2)) # Using above Formula
    return weights


def localWeight(point,xmat,ymat,k):
    wei = kernel(point,xmat,k)
    return (X.T*(wei*X)).I*(X.T*(wei*ymat.T))  # Calculate Beta(model term parameter) Using β(xo) = (X^T WX)^-1 X^T Wy



def localWeightRegression(xmat,ymat,k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    print(ypred)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred

data = pd.read_csv('AIML09_Regression_Algorithm.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)




mbill = np.mat(bill)
mtip = np.mat(tip)

m= np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T,mbill.T))


ypred = localWeightRegression(X,mtip,2)

SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,0]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(bill,tip)
ax.plot(xsort[:,1],ypred[SortIndex], color = 'red')
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()