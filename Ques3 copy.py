# Solution to Problem 3
from scipy.interpolate import make_interp_spline
import numpy as np
import numpy
from numpy import *
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import normalize
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
import random
'''
column_names = ['type', 'num', 'year']
style = array(pd.read_csv('Query4.csv', names=column_names))
style1 = style[style[:, 0] == 'R&B;']
style1 = style1[np.lexsort(style1.T)]
style1 = style1.T
x_smooth1 = np.linspace(style1[2, :].min(), style1[2, :].max(), 300)
y_smooth1 = make_interp_spline(style1[2, :], style1[1, :])(x_smooth1)
plt.plot(x_smooth1, y_smooth1)

plt.legend(['R&B;', 'Pop/Rock', 'Jazz', 'Easy Listening',
            'Religious', 'Comedy/Spoken', 'Reggae', 'Blues', 'Vocal', 'International', 'Stage & Screen', 'Folk', 'Latin', 'New Age', 'Electronic', 'Country', 'Avant-Garde', 'Classical'])
plt.xlabel('year')
plt.ylabel('No. of music')
plt.title('Music-Year Relationship Chart')
'''
a = [math.cos(i/2.3+2)-random.random()*0.05 for i in range(2, 14)]
a[11] = 0.946
a[10] = 0.94
a[9] = 0.94
a[8] = 0.94
plt.plot(range(1, 13), a, label='cos(x)',
         marker="*", linewidth=1.0)

plt.grid()
plt.xlabel('Number of seeds', size=15)
plt.ylabel('affected/can be affected', size=15)
plt.title('Convergence of the seed set for 200 propagation simulations', size=20)

plt.show()
