from lib.feature import *
from pprint import pprint

path = '../data/data_lite.txt'

# pmi
print('pmi')
pmi = MutualInfo(path)
print(pmi.text[0][:10]) # print text
pprint(pmi.X[0][:10])   # print string pmi

pmi = MutualInfo(path, uniform=False)
pprint(pmi.X[0][:10])   # print float pmi
