#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def pca(X, no_dims=50):
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def main():
    x = np.linspace(0, 5, 500)
    x = np.random.choice(x, 1000, replace=True)
    y = np.add(x,1)
    #y = np.add(y,generate_random(num=len(x)))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    coord = np.concatenate((x,y),axis=1)
    X_embedded = pca(coord)
    plt.scatter(X_embedded[:,0], X_embedded[:,1], 20)
    plt.show()

def generate_random(num):
    return np.random.normal(0, 1, num)*0.1

if __name__ == "__main__":
    main()
