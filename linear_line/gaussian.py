#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 0], [0, 100]]

mean2 = [100, 100]
cov2 = [[1, 0], [0, 100]]

x, y = np.random.multivariate_normal(mean, cov, 5000).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, 5000).T
plt.plot(x, y, "o", color="red")
plt.plot(x2, y2, "o", color="blue")
plt.axis("equal")
plt.show()

