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
from sklearn.ensemble import RandomForestClassifier
set_cuda_active(False)

def main():
    mat = scipy.io.loadmat("letter.mat")
    X = mat["X"]
    y = mat["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = RandomForestClassifier(max_depth=30, random_state=0)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(predictions)
    print(y_test.ravel())
    print(confusion_matrix(y_test.ravel(), predictions))
    print(classification_report(y_test.ravel(), predictions))
    #print("y_test:{}".format(len(y_test.ravel())))

if __name__ == "__main__":
    main()
