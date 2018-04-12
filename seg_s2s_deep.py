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

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

path = './data/data_proc.txt'
for k in range(9, 20):
    seg_size = 2*k + 1
    print('seg_size:', seg_size, file=open('lstm_seg.txt', 'a'))
    data = UniVec(path, k=0, vec_size=50)
    data.X = list(chunks(data.X, seg_size))[:-1]
    data.Y = [ [ele] for ele in data.Y ]
    data.Y = list(chunks(data.Y, seg_size))[:-1]
    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.Y, test_size=0.3, shuffle=False
    )
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print('start training')
    inputs = Input(shape=(len(x_test[0]), len(x_test[0][0])))
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inputs)
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
    ts = TensorBoard(log_dir='./log')
    model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop, ts], validation_split=0.1, epochs=100)
    pred = model.predict([x_test])
    union = lambda data: [ ins for chap in data for ins in chap]
    y_pred = data.y2lab(union(pred))
    y_test = data.y2lab(union(y_test))
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=('I', 'E'), digits=4
    ), file=open('lstm_seg.txt', 'a'))
