import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from lib.feature import *
from lib.metric import *
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from sklearn_crfsuite import metrics
import keras
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from keras.optimizers import Adam
from keras.layers import CuDNNLSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU, Bidirectional, CuDNNLSTM

train_path = './data/train.txt'
deep_k     = 10

deep_train = VecContext(train_path, k=deep_k, vec_size=50, w2v_text='./data/w2v.txt')

x_train = np.array(deep_train.X)
y_train = np.array(deep_train.Y)

inputs = Input(shape=(len(x_train[0]), len(x_train[0][0])))
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
model.save('./pickles/lstm-train.h5')
