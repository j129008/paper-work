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

path = './data/data_proc.txt'

for k in range(9, 15):
    print('context:', k, file=open('lstm_context.txt', 'a'))
    data = VecContext(path, k=k, vec_size=50)
    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.Y, test_size=0.3, shuffle=False
    )
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    inputs = Input(shape=(len(x_test[0]), len(x_test[0][0])))
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True, go_backwards=True))(inputs)
    for _ in range(3):
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
    ts = TensorBoard(log_dir='./log')
    model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop, ts], validation_split=0.1, epochs=100)
    pred = model.predict([x_test])
    y_pred = data.y2lab(pred)
    _y_test = data.y2lab(y_test)
    print(metrics.flat_classification_report(
        _y_test, y_pred, labels=('I', 'E'), digits=4
    ), file=open('lstm_context.txt', 'a'))
