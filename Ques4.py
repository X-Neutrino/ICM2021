# Solution to Problem 4
import seaborn as sns
from sklearn import datasets
import palettable
import numpy
from numpy import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import csv
import matplotlib.pyplot as plt
from pylab import mpl
import pandas as pd
from pandas import Series, DataFrame

total = 42771
style = array(pd.read_csv('type.csv', header=None))
music = array(pd.read_csv('full_music_data.csv'))
total = style[:, 1].sum(axis=0)
style[:, 1] = style[:, 1]/total

factor = music[:, 2:17]
popu = music[:, 15]
[rows, cols] = factor.shape
cc = np.zeros((rows, cols))
'''
for i in range(cols):
    cc = np.array([factor[:, i], popu])
    cc = cc.astype('float64')
    cc_mean = np.mean(cc, axis=0)  # axis=0,表示按列求均值 ——— 即第一维
    cc_std = np.std(cc, axis=0)
    cc_zscore = (cc-cc_mean)/cc_std  # 标准化
    cc_pd = pd.DataFrame(cc_zscore.T, columns=['c1', 'c2'])
    cc_corr = cc_pd.corr(method='spearman')  # 相关系数矩阵
    print(cc_corr['c1'][1])
'''
cc = np.array([factor[:, 6], popu])

cc = cc.astype('float64')
cc = cc.T
cc = np.row_stack((cc, [1.5, 700]))
cc = np.row_stack((cc, [-0.5, -500]))
cc = DataFrame(cc)
new_col = ['key', 'popularity']
cc.columns = new_col
plt.figure(dpi=100)
sns.set(style="whitegrid", font_scale=1.2)

g = sns.regplot(x='key', y='popularity', data=cc,
                color='#000000',
                marker='*',
                scatter_kws={'s': 10, 'color': 'g', },
                line_kws={'linestyle': '--', 'color': 'r'})
plt.show()
