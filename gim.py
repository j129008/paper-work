from gensim.models import Word2Vec
from pprint import pprint
from lib.learner import Learner
from lib.feature import Feature
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import metrics
import pickle

try:
    w2v = pickle.load(open('./pickles/word2vec.pkl', 'rb'))
except:
    sentence = open('./data/data2.txt', 'r').read().replace('\n','').split('，')
    sentence = [ list(ele) for ele in sentence]
    w2v = Word2Vec(sentence, min_count=1)
    pickle.dump(w2v, open('./pickles/word2vec.pkl', 'wb'))

def toValue(x, y):
    v_x = []
    v_y = []
    for ele in x:
        v_x.append( w2v[ele['0']] )
    for ele in y:
        if ele == 'E':
            v_y.append(1)
        else:
            v_y.append(0)
    return (v_x, v_y)

def to_label(y):
    labs = []
    for ele in y:
        if ele > 0.5:
            labs.append('E')
        else:
            labs.append('I')
    return labs

data = Learner('./data/data3.txt')
data.load_feature(funcs=[Feature.context], params=[{'k':0, 'n_gram':1}])
clf = RandomForestClassifier()
x, y = toValue(data.X_train, data.Y_train)
clf.fit(x, y)
x, y = toValue(data.X_private, data.Y_private)
Y_pred = clf.predict( x )

print(metrics.flat_classification_report(
    data.Y_private, to_label(Y_pred), labels=('I', 'E'), digits=4
))
