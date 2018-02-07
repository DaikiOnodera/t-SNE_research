#!/usr/bin/env python
# encoding:utf-8

import numpy as np
from sklearn.manifold import TSNE
X = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])
print("input:\n{}".format(X))
X_embedded = TSNE(n_components=2).fit_transform(X)
print("result shape:{}".format(X_embedded.shape))

