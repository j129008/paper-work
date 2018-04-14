import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from lib.feature import *
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate, RepeatVector
from sklearn_crfsuite import metrics
import keras
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from keras.optimizers import Adam
from keras.layers import CuDNNLSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU, Bidirectional, CuDNNLSTM
from sklearn.model_selection import train_test_split
from random import randint
from pprint import pprint

def chunks(l, n):
    for i in range(0, len(l)-len(l)%n, n):
        yield np.array(l[i:i + n])

def batch_gen(x, y, seq_len, batch_size=100):
    seq_cnt = len(y)//seq_len
    while True:
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            i = randint(0, seq_cnt-1)*seq_len
            seq_x = x[i:i+seq_len]
            seq_y = y[i:i+seq_len]
            batch_x.append(seq_x)
            batch_y.append(seq_y)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        yield (batch_x, batch_y)

path = './data/data_proc.txt'
data = UniVec(path, k=0, vec_size=50)
data.Y = [ [ele] for ele in data.Y ]

for k in range(9, 15):
    seg_size = 2*k + 1
    print('k:', k, file=open('lstm_seg.txt', 'a'))
    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.Y, test_size=0.3, shuffle=False
    )

    trans = lambda data : np.array(list(chunks(data, seg_size)))
    split_p = len(x_train)//10
    x_vaild = trans(x_train[-split_p:])
    y_vaild = trans(y_train[-split_p:])
    x_train = x_train[:-split_p]
    y_train = y_train[:-split_p]
    padding = (seg_size - len(x_test)%seg_size)%seg_size
    x_test = x_test + padding* [50*[0]]
    y_test = y_test + padding* [[0]]

    print('start training')
    inputs = Input(shape=(seg_size, 50))
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inputs)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Dropout(0.1)(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    main_output = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    model = Model(inputs=[inputs], outputs=main_output)
    keras.utils.plot_model(model, to_file='model.png')
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
    model.fit_generator(batch_gen(x_train, y_train, seq_len=seg_size, batch_size=100), steps_per_epoch=len(y_train)//1000, epochs=100, validation_data=(x_vaild, y_vaild), callbacks=[early_stop])
    pred = model.predict([trans(x_test)])
    union = lambda data: [ ins for chap in data for ins in chap]
    y_pred = data.y2lab(union(pred))
    y_test = data.y2lab(union(y_test))
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=('I', 'E'), digits=4
    ), file=open('lstm_seg.txt', 'a'))
