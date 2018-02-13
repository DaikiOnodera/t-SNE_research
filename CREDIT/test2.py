#!/usr/bin/env python
# encoding:utf-8

perplexity = 140.0
learning_rate = 20
eps = 0.3
#min_samples=5

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_mldata
from sklearn.cluster import DBSCAN

def pca(X, no_dims=50):
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def main():
    first_time = False
    if first_time:
        import os
        os.system("wget http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data")
    # data preprocessing
    df = pd.read_csv("crx.data", header=None, index_col=None)
    df = df.applymap(lambda d:np.nan if d=="?" else d)
    df = df.dropna(axis=0)
    sr_labels = df.iloc[:, -1]
    labels = sr_labels.str.replace("+","1").replace("-","0").values.astype(float)
    data = df.iloc[:, :-1].values.astype(str)

    # data transformation
    pattern_continuous = re.compile("^\d+\.?\d*\Z")
    continuous_idx = {}
    for i in range(data.shape[1]):
        is_continuous = True if pattern_continuous.match(data[0][i]) else False
        if is_continuous and i==0:
            X = data[:, i].astype(float)
        elif not is_continuous and i==0:
            X = pd.get_dummies(data[:, i]).values.astype(float)
        elif is_continuous and i!=0:
            X = np.concatenate((X, data[:, i].reshape(-1, 1).astype(float)), axis=1)
        elif not is_continuous and i!=0:
            X = np.concatenate((X, pd.get_dummies(data[:, i]).values.astype(float)), axis=1)
    print("X:{X.shape}, y:{labels.shape}".format(**locals()))
    1/0

    X = X
    y = labels
    X_PCA = pca(X, no_dims=30)
    X_PCA_real = pca(X, no_dims=30).real
    X_PCA_imag = pca(X, no_dims=30).imag
    print("X_PCA:\n{}".format(X_PCA))
    print("X_PCA_real:\n{}".format(X_PCA_real))
    print("X_PCA_imag:\n{}".format(X_PCA_imag))

    X_embedded_PCA = pca(X, no_dims=2)
    #X_embedded_TSNE = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=500, method="barnes_hut", verbose=2, y=y).fit_transform(X_PCA)
    X_embedded_TSNE = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=500, method="barnes_hut", verbose=2).fit_transform(X_PCA)

    for i in set(list(y.ravel())):
        idx = y.ravel() == i
        plt.scatter(X_embedded_PCA[idx, 0], X_embedded_PCA[idx, 1], color=cm.get_cmap("tab20").colors[int(i)], alpha=0.7)
    plt.legend()
    plt.savefig("PCA.png")
    plt.clf()

    for i in set(list(y.ravel())):
        idx = y.ravel() == i
        plt.scatter(X_embedded_TSNE[idx, 0], X_embedded_TSNE[idx, 1], color=cm.get_cmap("tab20").colors[int(i)], alpha=0.7)
    plt.legend()
    plt.savefig("TSNE.png")
    plt.clf()

    db = DBSCAN(eps=eps).fit(X_embedded_TSNE)
    y_db = db.labels_
    # outlier
    print(len(np.where(y_db==-1)[0]))
    # cluster
    for j in range(0, np.max(y_db)):
        plt.scatter(X_embedded_TSNE[np.where(y_db==j)[0], 0], X_embedded_TSNE[np.where(y_db==j)[0], 1], color=cm.get_cmap("Accent").colors[int(j)], label="cluster{j}".format(**locals()), alpha=0.7)
    plt.legend()
    plt.savefig("clusters.png".format(**locals()))

if __name__ == "__main__":
    main()
