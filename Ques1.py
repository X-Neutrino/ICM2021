# Solution to Problem 1
import numpy as np
import numpy
from numpy import *
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import normalize
import csv
import matplotlib.pyplot as plt
from pylab import mpl

# Set the Evaluation Matrix
A = array([[1, 5, 5, 7, 7], [1/5, 1, 1/3, 5, 3], [1/5, 3, 1, 5, 3],
           [1/7, 1/5, 1/5, 1, 1/3], [1/7, 1/3, 1/3, 3, 1]])

# Sorting Index
r = array(A.sum(axis=1))
rmax = max(r)
rmin = min(r)
k = rmax/rmin

# Building the Evaluation Matrix
B = np.zeros((5, 5))
[rows, cols] = B.shape
for i in range(rows):
    for j in range(cols):
        if(r[i] >= r[j]):
            B[i, j] = (r[i]-r[j])/(rmax-rmin)*(k-1)+1
        else:
            B[i, j] = 1/((r[i]-r[j])/(rmax-rmin)*(k-1)+1)
print("B=", B)

# Calculatting the Fitted Agreement Matrix
tmp = array(B.sum(axis=1))
print(tmp)
C = np.zeros((5, 5))
[rows, cols] = C.shape
for i in range(rows):
    for j in range(cols):
        C[i, j] = tmp[i]-tmp[j]
C = (10**(1/4))*C
print("C=", C)

# Calculatting the Weights


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x


Cmax = C.max()
Cmin = C.min()
C = MaxMinNormalization(C, Cmax, Cmin)
W = array(C.sum(axis=1)/5)
print("W=", W)

# load full_music_data
person = array(pd.read_csv('Query1.csv', header=None))
per = array(pd.read_csv('Query1.csv', header=None)[0])
year = array(pd.read_csv('Query2.csv', header=None))
style = array(pd.read_csv('Query3.csv', header=None))

[rows, cols] = person.shape
[rows2, cols2] = year.shape
[rows3, cols3] = style.shape
Iflunce = np.zeros([rows, 1])
per = per.reshape(-1, 1)
Iflunce = hstack((per, Iflunce))

total = 3771
yeap = 0
# Calculate the value of influence
for i in range(rows):
    Iflunce[i, 1] = person[i, 3]*W[0]+person[i, 4]*W[1]
    for j in range(rows2):
        if(year[j, 0] == person[i, 1]):
            Iflunce[i, 1] = Iflunce[i, 1]+(year[j, 1]+year[j, 2])/total*W[2]
    for k in range(rows3):
        if(style[k, 0] == person[i, 2]):
            yeap = 0
            for k2 in range(rows2):
                if(style[k, 1] == year[j, 0]):
                    yeap = yeap+year[j, 1]+year[j, 2]
            if(yeap != 0):
                Iflunce[i, 1] = Iflunce[i, 1]+(style[j, 2])/yeap*W[3]
Iflunce = Iflunce[np.argsort(Iflunce[:, 1]), :]
Iflunce = np.flipud(Iflunce)
print(Iflunce)
data1 = DataFrame(Iflunce)
data1.to_csv('Iflunce.csv', index=False, header=None)

xlab = Iflunce[0:10, 0]
ylab = Iflunce[0:10, 1]

# Extract the top ten scores and draw the bar graph


def draw(x_data, y_data, title, xytitle, is_showval):
    plt.figure()
    bar_width = 0.3  # width
    plt.bar(x=x_data, height=y_data, color="w", edgecolor="k",
            alpha=0.8, width=bar_width, hatch="\\\\")
    if(is_showval):  # whether to show value
        for x, y in enumerate(y_data):
            plt.text(x, y, '%.2f' % y, ha='center', va='bottom')
    plt.title(title, fontsize=20)
    plt.xlabel(xytitle[0], fontsize=18)
    plt.ylabel(xytitle[1], fontsize=18)
    plt.grid(alpha=0.8, linestyle='-.', axis='y')
    plt.show()


ydata = ylab
xdata = xlab
xytitle = ['Music figures or groups',
           'Influence Value']
ptitle = 'The most influential music personalities(TOP 10)'
is_showval = 1  # show value
draw(xdata, ydata, ptitle, xytitle, is_showval)
