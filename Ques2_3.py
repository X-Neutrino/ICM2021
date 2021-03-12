import matplotlib.pyplot as plt
import numpy as np
import numpy
from numpy import *
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import datasets
import seaborn as sns
import palettable

column_names = ['aname', 'aid', 'danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode', 'key', 'acousticness',
                'instrumentalness', 'liveness', 'speechiness', 'explicit', 'duration', 'popularity', 'year', 'releasedate', 'title', 'clu', 'type']
clust = pd.read_csv('clust2.csv', names=column_names)
clust = clust.dropna()
clust.drop_duplicates(inplace=True)
clust = array(clust)
clust1 = clust[clust[:, 19] == 0]
clust2 = clust[clust[:, 19] == 1]
clust3 = clust[clust[:, 19] == 2]
clust4 = clust[clust[:, 19] == 3]
clust5 = clust[clust[:, 19] == 4]
clust6 = clust[clust[:, 19] == 5]

x = clust6[:, [2, 5]]
y = clust6[:, 20]
clust6 = np.concatenate((x, y[:, None]), axis=1)
clust6 = DataFrame(clust6)
new_col = ['danceability', 'tempo', 'type']
clust6.columns = new_col
for col in ['danceability', 'tempo']:
    clust6[col] = clust6[col].astype('float64')
g = sns.pairplot(clust6,
                 hue='type',

                 )
sns.set(style='whitegrid')
g = g.map_upper(sns.scatterplot, linewidth=0, s=0.01)
g = g.map_lower(sns.scatterplot, linewidth=0, s=0.01)
g.fig.set_size_inches(12, 12)
sns.set(style='whitegrid', font_scale=1.5)
plt.show()
