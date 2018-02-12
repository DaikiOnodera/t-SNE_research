#!/usr/bin/env python
# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 0], [0, 100]]

x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, "o")
plt.axis("equal")
plt.show()
