import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# This code file is used for data pre-processing
# Set columns' name
column_names = ['Iid', 'Iname', 'Igenre',
                'Iactive', 'Fid', 'Fname', 'Fgenre', 'Factive']
df = pd.read_csv('influence_data.csv', names=column_names)

# Delete non-ASCII characters
df['Iname'].replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
df['Fname'].replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

# Delete empty lines
df.dropna(how='all', inplace=True)

# Delete duplicate rows
df.drop_duplicates(inplace=True)
df = df.drop(labels=0)
df.to_csv('new.csv', index=False)

# Virtualized column variables(ONE-HOT)
df = df.drop(['Iname', 'Fname'], axis=1)
df = pd.get_dummies(df, columns=['Igenre', 'Fgenre'],
                    prefix_sep='_')

min_max = MinMaxScaler()
df = pd.DataFrame(min_max.fit_transform(df))  # 对数值变量min_max归一化
# Save the pre-processed data
df.to_csv('new2.csv', index=False)
