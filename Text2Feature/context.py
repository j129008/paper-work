from lib.feature import *
from pprint import pprint

path = '../data/data_lite.txt'

# for crf
data = Context(path, k=1, n_gram=2)
print(data.text[0][:10]) # print text
pprint(data.X[0][:10])   # print feature

# for lstm
data = VecContext(path, k=1, w2v_text='../data/w2v.txt', vec_size=5)
print(data.text[:3]) # print text
pprint(data.X[:3])   # print feature

# change w2v dim
data = VecContext(path, k=1, w2v_text='../data/w2v.txt', vec_size=10)
print(data.text[:3]) # print text
pprint(data.X[:3])   # print feature

# for other model
data = UniVec(path, k=1, vec_size=10)
print(data.text[:3]) # print text
pprint(data.X[:3])   # print feature
