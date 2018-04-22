import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from lib.feature import *
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from sklearn_crfsuite import metrics
import keras
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from keras.optimizers import Adam
from keras.layers import CuDNNLSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU, Bidirectional, CuDNNLSTM
from sklearn.model_selection import train_test_split
import re

path = './data/data_proc.txt'
k = 12
data = VecContext(path, k=k, vec_size=50, w2v_text='./data/w2v.txt')
cutter = int(len(data.Y)*0.3)
x_train = np.array(data.X[cutter:])
x_test = np.array(data.X[:cutter])
y_train = np.array(data.Y[cutter:])
y_test = np.array(data.Y[:cutter])

inputs = Input(shape=(len(x_test[0]), len(x_test[0][0])))
x = Bidirectional(CuDNNLSTM(50, return_sequences=True, go_backwards=True))(inputs)
for _ in range(4):
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True, go_backwards=True))(x)
x= Bidirectional(CuDNNLSTM(50))(x)
main_output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[inputs], outputs=main_output)
keras.utils.plot_model(model, to_file='model.png')
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.01, epochs=100)
model.save('./pickles/keras.h5')
pred = model.predict([x_test])
y_pred = data.y2lab(pred)
y_test = data.y2lab(y_test)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=('I', 'E'), digits=4
))

f = open('deep_err.txt', 'w')
i = 0
miss_list = []
err_list = []
for line in data.text:
    T = ''
    P = ''
    if i > len(y_pred)-1:
        break
    for char in line.strip():
        if i > len(y_pred)-1:
            break
        T += char
        P += char
        if y_pred[i] != y_test[i]:
            if y_pred[i] == 'I':
                P += '　'
                T += '，'
            else:
                P += '，'
                T += '　'
        else:
            if y_pred[i] == 'E':
                P += '，'
                T += '，'
        i+=1
    f.write('T: '+T+'\n')
    f.write('P: '+P+'\n')
    miss_list.extend(re.findall(r'(...)　', P))
    err_list.extend(re.findall(r'(...)　', T))
f.write('miss list:\n' + '\n'.join([ str(ele) for ele in Counter(miss_list).most_common(10) ]) + '\n')
f.write('err list:\n' + '\n'.join([ str(ele) for ele in Counter(err_list).most_common(10) ]) + '\n')
