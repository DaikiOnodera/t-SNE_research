#!/usr/bin/env python
# encoding:utf-8

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

N = 50
fig, ax = plt.subplots()

def update(i):
    a = np.random.uniform(0, 255, [N, N, 3])
    a[i:, i:] = 0.
    plt.clf()
    plf.imshow(a)

animation.FuncAnimation(fig, update, np.arange(1, N), interval=25)
plt.show()
