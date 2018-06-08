from lib.feature import *
from pprint import pprint

path = '../data/data_lite.txt'

pmi = MutualInfo(path)
print(pmi.text[0][:10]) # print text
pprint(pmi.X[0][:10])   # print string pmi

data = Context(path)
pprint(data.X[0][:10])   # print string pmi

pprint((data+pmi).X[0][:10])   # print string pmi
