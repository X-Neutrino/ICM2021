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

column_names = ['type', 'num', 'year']
style = array(pd.read_csv('Query4.csv', names=column_names))
style1 = style[style[:, 0] == 'R&B;']
style1 = style1[np.lexsort(style1.T)]
style1 = style1.T
x_smooth1 = np.linspace(style1[2, :].min(), style1[2, :].max(), 300)
y_smooth1 = make_interp_spline(style1[2, :], style1[1, :])(x_smooth1)
plt.plot(x_smooth1, y_smooth1)

style2 = style[style[:, 0] == 'Pop/Rock']
style2 = style2[np.lexsort(style2.T)]
style2 = style2.T
x_smooth2 = np.linspace(style2[2, :].min(), style2[2, :].max(), 300)
y_smooth2 = make_interp_spline(style2[2, :], style2[1, :])(x_smooth2)
plt.plot(x_smooth2, y_smooth2)

style3 = style[style[:, 0] == 'Jazz']
style3 = style3[np.lexsort(style3.T)]
style3 = style3.T
x_smooth3 = np.linspace(style3[2, :].min(), style3[2, :].max(), 300)
y_smooth3 = make_interp_spline(style3[2, :], style3[1, :])(x_smooth3)
plt.plot(x_smooth3, y_smooth3)

style4 = style[style[:, 0] == 'Easy Listening']
style4 = style4[np.lexsort(style4.T)]
style4 = style4.T
x_smooth4 = np.linspace(style4[2, :].min(), style4[2, :].max(), 300)
y_smooth4 = make_interp_spline(style4[2, :], style4[1, :])(x_smooth4)
plt.plot(x_smooth4, y_smooth4)

style5 = style[style[:, 0] == 'Religious']
style5 = style5[np.lexsort(style5.T)]
style5 = style5.T
x_smooth5 = np.linspace(style5[2, :].min(), style5[2, :].max(), 300)
y_smooth5 = make_interp_spline(style5[2, :], style5[1, :])(x_smooth5)
plt.plot(x_smooth5, y_smooth5)

style6 = style[style[:, 0] == 'Comedy/Spoken']
style6 = style6[np.lexsort(style6.T)]
style6 = style6.T
x_smooth6 = np.linspace(style6[2, :].min(), style6[2, :].max(), 300)
y_smooth6 = make_interp_spline(style6[2, :], style6[1, :])(x_smooth6)
plt.plot(x_smooth6, y_smooth6)

style7 = style[style[:, 0] == 'Reggae']
style7 = style7[np.lexsort(style7.T)]
style7 = style7.T
x_smooth7 = np.linspace(style7[2, :].min(), style7[2, :].max(), 300)
y_smooth7 = make_interp_spline(style7[2, :], style7[1, :])(x_smooth7)
plt.plot(x_smooth7, y_smooth7)

style8 = style[style[:, 0] == 'Blues']
style8 = style8[np.lexsort(style8.T)]
style8 = style8.T
x_smooth8 = np.linspace(style8[2, :].min(), style8[2, :].max(), 300)
y_smooth8 = make_interp_spline(style8[2, :], style8[1, :])(x_smooth8)
plt.plot(x_smooth8, y_smooth8)

style9 = style[style[:, 0] == 'Vocal']
style9 = style9[np.lexsort(style9.T)]
style9 = style9.T
x_smooth9 = np.linspace(style9[2, :].min(), style9[2, :].max(), 300)
y_smooth9 = make_interp_spline(style9[2, :], style9[1, :])(x_smooth9)
plt.plot(x_smooth9, y_smooth9)

style10 = style[style[:, 0] == 'International']
style10 = style10[np.lexsort(style10.T)]
style10 = style10.T
x_smooth10 = np.linspace(style10[2, :].min(), style10[2, :].max(), 300)
y_smooth10 = make_interp_spline(style10[2, :], style10[1, :])(x_smooth10)
plt.plot(x_smooth10, y_smooth10)

style11 = style[style[:, 0] == 'Stage & Screen']
style11 = style11[np.lexsort(style11.T)]
style11 = style11.T
x_smooth11 = np.linspace(style11[2, :].min(), style11[2, :].max(), 300)
y_smooth11 = make_interp_spline(style11[2, :], style11[1, :])(x_smooth11)
plt.plot(x_smooth11, y_smooth11)

style12 = style[style[:, 0] == 'Folk']
style12 = style12[np.lexsort(style12.T)]
style12 = style12.T
x_smooth12 = np.linspace(style12[2, :].min(), style12[2, :].max(), 300)
y_smooth12 = make_interp_spline(style12[2, :], style12[1, :])(x_smooth12)
plt.plot(x_smooth12, y_smooth12)

style13 = style[style[:, 0] == 'Latin']
style13 = style13[np.lexsort(style13.T)]
style13 = style13.T
x_smooth13 = np.linspace(style13[2, :].min(), style13[2, :].max(), 300)
y_smooth13 = make_interp_spline(style13[2, :], style13[1, :])(x_smooth13)
plt.plot(x_smooth13, y_smooth13)

style14 = style[style[:, 0] == 'New Age']
style14 = style14[np.lexsort(style14.T)]
style14 = style14.T
x_smooth14 = np.linspace(style14[2, :].min(), style14[2, :].max(), 300)
y_smooth14 = make_interp_spline(style14[2, :], style14[1, :])(x_smooth14)
plt.plot(x_smooth14, y_smooth14)

style15 = style[style[:, 0] == 'Electronic']
style15 = style15[np.lexsort(style15.T)]
style15 = style15.T
x_smooth15 = np.linspace(style15[2, :].min(), style15[2, :].max(), 300)
y_smooth15 = make_interp_spline(style15[2, :], style15[1, :])(x_smooth15)
plt.plot(x_smooth15, y_smooth15)

style16 = style[style[:, 0] == 'Country']
style16 = style16[np.lexsort(style16.T)]
style16 = style16.T
x_smooth16 = np.linspace(style16[2, :].min(), style16[2, :].max(), 300)
y_smooth16 = make_interp_spline(style16[2, :], style16[1, :])(x_smooth16)
plt.plot(x_smooth16, y_smooth16)

style17 = style[style[:, 0] == 'Avant-Garde']
style17 = style17[np.lexsort(style17.T)]
style17 = style17.T
x_smooth17 = np.linspace(style17[2, :].min(), style17[2, :].max(), 300)
y_smooth17 = make_interp_spline(style17[2, :], style17[1, :])(x_smooth17)
plt.plot(x_smooth17, y_smooth17)

style18 = style[style[:, 0] == 'Classical']
style18 = style18[np.lexsort(style18.T)]
style18 = style18.T
x_smooth18 = np.linspace(style18[2, :].min(), style18[2, :].max(), 300)
y_smooth18 = make_interp_spline(style18[2, :], style18[1, :])(x_smooth18)
plt.plot(x_smooth18, y_smooth18)

plt.legend(['R&B;', 'Pop/Rock', 'Jazz', 'Easy Listening',
            'Religious', 'Comedy/Spoken', 'Reggae', 'Blues', 'Vocal', 'International', 'Stage & Screen', 'Folk', 'Latin', 'New Age', 'Electronic', 'Country', 'Avant-Garde', 'Classical'])
plt.xlabel('year')
plt.ylabel('No. of music')
plt.title('Music-Year Relationship Chart')
plt.show()
