from gensim.models import Word2Vec
from pprint import pprint

sentence = open('./data/data2.txt', 'r').read().replace('\n','').split('，')
sentence = [ list(ele) for ele in sentence]
model = Word2Vec(sentence)
print(model['女'])
