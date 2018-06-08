from lib.feature import *
from pprint import pprint

path = '../data/data_lite.txt'

# tdiff
print('tdiff')
tdiff = MutualInfo(path)
print(tdiff.text[0][:10]) # print text
pprint(tdiff.X[0][:10])   # print string pmi

tdiff = Tdiff(path, uniform=False)
pprint(tdiff.X[0][:10])   # print float pmi
