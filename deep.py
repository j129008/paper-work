import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn_crfsuite import metrics
import pickle
from keras.optimizers import Adam
from keras.layers import LSTM, TimeDistributed, SimpleRNN, Embedding, RNN

def to_label(y):
    labs = []
    for ele in y:
        if ele > 0.5:
            labs.append('E')
        else:
            labs.append('I')
    return labs

# Generate dummy data
x_train = pickle.load(open('./pickles/_vec_x_train.pkl','rb'))
dim = len(x_train[0])
print(dim)
x_train = np.array(x_train)
x_test  = np.array(pickle.load(open('./pickles/_vec_x_test.pkl', 'rb')))
y_test  = np.array(pickle.load(open('./pickles/_vec_y_test.pkl','rb')))
y_train = np.array(pickle.load(open('./pickles/_vec_y_train.pkl','rb')))

model = Sequential()
model.add(Embedding(64, output_dim=64))
model.add(LSTM(64))
#  model.add(Dense(64, activation='sigmoid', input_dim=1500))
#  model.add(Dense(64, activation='sigmoid', input_dim=1500))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=3)
pred = model.predict(x_test)
Y_private = to_label(y_test)
Y_pred = to_label(pred)
print(metrics.flat_classification_report(
    Y_private, Y_pred, labels=('I', 'E'), digits=4
))
