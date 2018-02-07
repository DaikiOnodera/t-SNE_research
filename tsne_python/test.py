#!/usr/bin/env python
# encoding:utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

n = np.linspace(-100.0, 100.0, 10000)

p = []
for i in range(len(n)):
    p.append(norm.pdf(x=n[i], loc=5, scale=100))

plt.scatter(n, p)
plt.show()
