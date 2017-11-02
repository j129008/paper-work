import pandas as pd
import pickle
from tqdm import tqdm as bar

df = pd.read_html('http://ytenx.org/kyonh/sieux?page='+str(1))[0]
df.columns = df.iloc[0]
df = df.drop(0, 0)
df_list = [df]

for i in bar( range(2, 260) ):
    data = pd.read_html('http://ytenx.org/kyonh/sieux?page='+str(i))[0]
    data.columns = data.iloc[0]
    data = data.drop(0, 0)
    df_list.append(data)

df = pd.concat(df_list)
df = df.drop('次序', 1)
df = df.T
df.columns = df.iloc[0]
df = df.drop('小韻', 0)
pickle.dump(df, open('./pickles/rhyme_list.pkl', 'wb'))
