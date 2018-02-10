import numpy as np
from gensim.models import Word2Vec
import pickle
from lib.learner import Learner
from lib.feature import Feature
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn_crfsuite import metrics
from keras.optimizers import Adam
from keras.layers import LSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU, Bidirectional
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

try:
    w2v = pickle.load(open('./pickles/word2vec.pkl', 'rb'))
except:
    sentence = open('./data/data2.txt', 'r').read().replace('\n','').split('，')
    sentence = [ list(ele) for ele in sentence]
    w2v = Word2Vec(sentence, min_count=1, workers=8, iter=50)
    pickle.dump(w2v, open('./pickles/word2vec.pkl', 'wb'))

path = './data/data2.txt'
voc_size = len(set(open(path, 'r').read()))

def x2list(x):
    docs = []
    for ele in x:
        instance = [ w2v[ele[str(i)]] for i in range(len(ele)) ]
        docs.append(instance)
    return np.array(docs)
def y2bin(y):
    return np.array([ 1 if ele == 'E' else 0 for ele in y ])
def y2lab(y):
    return np.array([ 'E' if ele > 0.5 else 'I' for ele in y ])

data = Learner(path)
k = 4
data.load_feature(funcs=[Feature.context], params=[{'k':k, 'n_gram':1}])
x_train = x2list(data.X_train)
x_test = x2list(data.X_private)
y_train = y2bin(data.Y_train)

model = Sequential()
#  model.add(Embedding(voc_size, output_dim=100, input_length=3))
for _ in range(4):
    model.add(Bidirectional(LSTM(50, return_sequences=True, go_backwards=True), input_shape=(k*2+1, 100)))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=5)
pred = model.predict(x_test)
Y_private = data.Y_private
Y_pred = y2lab(pred)
print(metrics.flat_classification_report(
    Y_private, Y_pred, labels=('I', 'E'), digits=4
))

f = open('./pred.txt', 'w')
cutter = 20
line_true = ''
line_pred = ''
for i in range(len(data.Y_private)):
    word = data.X_private[i][str(k)]
    if i%cutter == 0:
        f.write(line_true+'\n')
        f.write(line_pred+'\n')
        f.write('\n')
        line_true = ''
        line_pred = ''
    line_pred+=word
    line_true+=word
    pred_w = Y_pred[i]
    real_w = Y_private[i]
    if pred_w == real_w and real_w == 'E':
        line_pred+='，'
        line_true+='，'
    if pred_w == 'E' and real_w == 'I':
        line_pred+='，'
        line_true+='　'
    if pred_w == 'I' and real_w == 'E':
        line_pred+='　'
        line_true+='，'
