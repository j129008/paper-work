import pickle
import pandas as pd

db = pickle.load(open('./pickles/rhyme_list.pkl', 'rb'))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(db.T)
