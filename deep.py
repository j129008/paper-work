import numpy as np
from lib.learner import Learner
from lib.feature import Feature
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn_crfsuite import metrics
from keras.optimizers import Adam
from keras.layers import LSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

path = './data/data2.txt'
voc_size = len(set(open(path, 'r').read()))

def x2list(x):
    docs = []
    for ele in x:
        docs.append(' '.join([ele['0'], ele['1'], ele['2']]))
    enc_docs = [ one_hot(d, voc_size) for d in docs ]
    padded_docs = pad_sequences(enc_docs, maxlen=3, padding='post')
    return np.array(padded_docs)
def y2bin(y):
    return np.array([ 1 if ele == 'E' else 0 for ele in y ])
def y2lab(y):
    return np.array([ 'E' if ele > 0.5 else 'I' for ele in y ])

data = Learner(path)
data.load_feature(funcs=[Feature.context], params=[{'k':1, 'n_gram':1}])
x_train = x2list(data.X_train)
x_test = x2list(data.X_private)
y_train = y2bin(data.Y_train)

model = Sequential()
model.add(Embedding(voc_size, output_dim=100, input_length=3))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=10, epochs=10)
pred = model.predict(x_test)
Y_private = data.Y_private
Y_pred = y2lab(pred)
print(metrics.flat_classification_report(
    Y_private, Y_pred, labels=('I', 'E'), digits=4
))
