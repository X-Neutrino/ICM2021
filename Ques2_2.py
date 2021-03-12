import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
pmusic = pd.read_csv('full_music_data.csv')
pmusic['artist_names'].replace(
    {r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
pmusic['release_date'].replace(
    {r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
pmusic['song_title (censored)'].replace(
    {r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
music = pmusic.dropna()
music.drop_duplicates(inplace=True)
music = music.drop(
    ['artist_names', 'artists_id', 'release_date', 'song_title (censored)'], axis=1)

music = music[['danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode', 'key',
               'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'explicit', 'duration_ms', 'popularity', 'year']]
music = (music - music.mean(axis=0))/(music.std(axis=0))  # z-score
kmodel = KMeans(n_clusters=6, n_jobs=8)
kmodel.fit(music)  # training model

label = pd.Series(kmodel.labels_)
num = pd.Series(kmodel.labels_).value_counts()
center = pd.DataFrame(kmodel.cluster_centers_)  # find center
max = center.values.max()
min = center.values.min()

X = pd.concat([center, num], axis=1)
X.columns = list(music.columns) + ['NUM']

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)  # polar=True
feature = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode', 'key',
           'acousticness', 'instrumentalness', 'liveness', 'speechiness', 'explicit', 'duration', 'popularity', 'year']
center_num = X.values  # <class 'numpy.ndarray'>
N = len(feature)

# plot
for i, v in enumerate(center_num):

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    center = np.concatenate((v[:-1], [v[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    ax.plot(angles, center, 'o-', linewidth=2, label="NO.%d = %d (%d%%)" %
            (i + 1, v[-1], v[-1]*100/93500))

    ax.fill(angles, center, alpha=0.25)

    ax.set_thetagrids(angles * 180 / np.pi, feature, fontsize=12)

    ax.set_ylim(min - 0.1, max + 0.1)

    plt.title('Music indicators', fontsize=20)

    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(
        1.3, 1.0), ncol=1, fancybox=True, shadow=True)

# save
plt.show()
shop_data_output = 'OUT2.csv'
out = pd.concat([pmusic, label], axis=1)
out.to_csv(shop_data_output, index=False)
