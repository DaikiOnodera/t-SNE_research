#!/usr/bin/env python
# encoding:utf-8

import numpy as np
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
    mat = scipy.io.loadmat("letter.mat")
    X = mat["X"]
    y = mat["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sequential = rm.Sequential([
        rm.Dense(32),
        rm.Relu(),
        rm.Dense(16),
        rm.Relu(),
        rm.Dense(1)
    ])
    batch_size = 128
    epoch = 500
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
