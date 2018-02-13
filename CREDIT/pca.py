#!/usr/bin/env python
# encoding:utf-8

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
from sklearn.manifold import TSNE

import renom as rm
from renom.optimizer import Adam
from renom.cuda import set_cuda_active
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
set_cuda_active(False)

def main():
    df = pd.read_csv("crx.data", header=None, index_col=None)
    df = df.applymap(lambda d:np.nan if d=="?" else d)
    df = df.dropna(axis=0)
    sr_labels = df.iloc[:, -1]
    labels = sr_labels.str.replace("+","1").replace("-","0").values.astype(float)
    data = df.iloc[:, :-1].values.astype(str)
    
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

    X = X
    y = labels.reshape(-1, 1)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.savefig("pca_credit.png")

if __name__ == "__main__":
    main()
