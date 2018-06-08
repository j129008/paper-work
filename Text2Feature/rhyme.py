from lib.feature import *
from pprint import pprint

path = '../data/data_lite.txt'
type_list = ['反切', '聲母', '韻目', '調', '等', '呼', '韻母']
rhyme = Rhyme(path, '../data/rhyme.txt', '../pickles/rhyme_list.pkl', type_list)

for t in type_list:
    t_data = rhyme.get_feature(t)
    pprint(t_data.X[0][:10])
