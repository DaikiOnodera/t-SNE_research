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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
set_cuda_active(False)

class AutoEncoder(rm.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.layer1 = rm.Dense(2)
        self.layer2 = rm.Dense(46)
    def forward(self, X):
        t1 = self.layer1(X)
        out = self.layer2(t1)
        return out
    def visualize(self, X):
        t1 = self.layer1(X)
        plt.scatter(t1[:, 0], t1[:, 1], alpha=0.7)
        plt.legend()
        plt.savefig("autoencoder_credit.png")

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

    model = AutoEncoder()
    batch_size = 128
    epoch = 50
    N = len(X)
    optimizer = Adam()
    for i in range(epoch):
        perm = np.random.permutation(N)
        loss = 0
        for j in range(0, N//batch_size):
            train_batch = X[perm[j*batch_size:(j+1)*batch_size]]
            with model.train():
                l = rm.mse(model(train_batch), train_batch)
            grad = l.grad()
            grad.update(optimizer)
            loss += l.as_ndarray()
        train_loss = loss / (N // batch_size)
        print("epoch:{:03d}, train_loss:{:.4f}".format(i, float(train_loss)))
    model.visualize(X)

if __name__ == "__main__":
    main()
