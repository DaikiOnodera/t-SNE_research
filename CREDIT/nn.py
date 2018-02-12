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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sequential = rm.Sequential([
        rm.Dense(46),
        rm.Relu(),
        rm.Dense(16),
        rm.Relu(),
        rm.Dense(1)
    ])
    batch_size = 128
    epoch = 50
    N = len(X_train)
    optimizer = Adam()
    for i in range(epoch):
        perm = np.random.permutation(N)
        loss = 0
        for j in range(0, N//batch_size):
            train_batch = X_train[perm[j*batch_size:(j+1)*batch_size]]
            response_batch = y_train[perm[j*batch_size:(j+1)*batch_size]]
            with sequential.train():
                l = rm.sgce(sequential(train_batch), response_batch)
            grad = l.grad()
            grad.update(optimizer)
            loss += l.as_ndarray()
        train_loss = loss / (N // batch_size)
        test_loss = rm.sgce(sequential(X_test), y_test).as_ndarray()
        print("epoch:{:03d}, train_loss:{:.4f}, test_loss:{:.4f}".format(i, float(train_loss), float(test_loss)))
    predictions = np.argmax(sequential(X_test).as_ndarray(), axis=1)
    print(confusion_matrix(y_test.ravel(), predictions))
    print(classification_report(y_test.ravel(), predictions))
    #print("y_test:{}".format(len(y_test.ravel())))

if __name__ == "__main__":
    main()
