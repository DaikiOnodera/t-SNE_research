#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
from sklearn.manifold import TSNE

def pca(X, no_dims=50):
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def main():
    mat = scipy.io.loadmat("letter.mat")
    X = mat["X"]
    y = mat["y"]
    X_PCA = pca(X, no_dims=30)
    X_embedded_PCA = pca(X, no_dims=2)
    X_embedded_TSNE = TSNE(n_components=2, perplexity=100.0, learning_rate=200, n_iter=5000, method="barnes_hut", verbose=2, y=y).fit_transform(X_PCA)

    for i in set(list(y.ravel())):
        idx = y.ravel() == i
        plt.scatter(X_embedded_PCA[idx, 0], X_embedded_PCA[idx, 1], color=cm.get_cmap("tab20").colors[i], alpha=0.7)
    plt.legend()
    plt.savefig("PCA.png")
    plt.clf()

    for i in set(list(y.ravel())):
        idx = y.ravel() == i
        plt.scatter(X_embedded_TSNE[idx, 0], X_embedded_TSNE[idx, 1], color=cm.get_cmap("tab20").colors[i], alpha=0.7)
    plt.legend()
    plt.savefig("TSNE.png")
    plt.clf()

    #print(X[idx, :].shape)
    #print(set(list(y.ravel())))
    #print(len(set(y)))

if __name__ == "__main__":
    main()
