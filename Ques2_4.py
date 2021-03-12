from sklearn.preprocessing import MinMaxScaler
import numpy as np
import numpy
from numpy import *
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import datasets
pmusic = pd.read_csv('data_by_artist.csv')
pmusic['artist_name'].replace(
    {r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
music = pmusic.dropna()
music.drop_duplicates(inplace=True)
music = music.drop(
    ['artist_name', 'artist_id'], axis=1)
music = array(music)
music = music[0:100, :]


def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b)/(a_norm * b_norm)
    return cos


cmat = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        cmat[i, j] = np.sqrt(np.sum(np.square(music[i]-music[j])))
min_max = MinMaxScaler()

cmat = DataFrame(cmat)
cmat = pd.DataFrame(min_max.fit_transform(cmat))
print(cmat)
cmat.to_csv('CosMatrix.csv', index=False, header=None)
