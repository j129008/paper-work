from lib.feature import *
from pprint import pprint

path = '../data/data_lite.txt'
#  string type
type_list = ['反切', '聲母', '韻目', '調', '等', '呼', '韻母']
rhyme = Rhyme(path, '../data/rhyme.txt', '../pickles/rhyme_list.pkl', type_list)
data = Context(path, k=0)

for t in type_list:
    data += rhyme.get_feature(t)
pprint(data.X[0][:10])

# digit type
type_list = ['調']
rhyme = VecRhyme(path, '../data/rhyme.txt', '../pickles/rhyme_list.pkl', type_list)
data = Context(path, k=0)

for t in type_list:
    data += rhyme.get_feature(t)
pprint(len(data.X[0][0]['調']))
