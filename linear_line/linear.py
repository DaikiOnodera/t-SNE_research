#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main():
    x = np.linspace(0, 5, 500)
    x = np.random.choice(x, 300, replace=True)
    y = np.add(x,1)
    #y = np.add(y,generate_random(num=len(x)))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    coord = np.concatenate((x,y),axis=1)
    print(coord.shape)
    plt.scatter(coord[:,0],coord[:,1],20)
    plt.savefig("images/original/original.png")
    X_embedded = TSNE(n_components=2, perplexity=100.0, learning_rate=10, n_iter=5000, method="exact", verbose=2).fit_transform(coord)
    plt.clf()
    plt.scatter(X_embedded[:,0], X_embedded[:,1], 20)
    plt.show()

def generate_random(num):
    return np.random.normal(0, 1, num)*0.1

if __name__ == "__main__":
    main()
