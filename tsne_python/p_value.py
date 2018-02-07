#!/usr/bin/env python
# encoding:utf-8

from pandas import DataFrame
from scipy import stats
import numpy as np

data = {"国語": [68, 75, 80, 71, 73, 79, 69, 65],
        "数学": [86, 83, 76, 81, 75, 82, 87, 75],
        "理科": [85, 69, 77, 77, 75, 74, 87, 69],
        "英語": [80, 76, 84, 93, 76, 80, 79, 84]}

df = DataFrame(data, index = ["生徒" + str(i+1) for i in np.arange(8)])
t, p = stats.ttest_rel(df["国語"], df["数学"])
print("locals:{}".format(locals()))
#print("p値=%(p)s" % locals())
