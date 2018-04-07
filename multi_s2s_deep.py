import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from lib.feature import *
from lib.data import Data
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate, RepeatVector
from sklearn_crfsuite import metrics
import keras
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from keras.optimizers import Adam
from keras.layers import CuDNNLSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU, Bidirectional, CuDNNLSTM
from sklearn.model_selection import train_test_split

path = './data/data_proc.txt'
k = 5
data = VecContext(path, k=k, vec_size=50, w2v_text='./data/w2v.txt')
seq_y = []
y = [0]*k + list(data.Y) + [0]*k
for i in range(k, len(data.Y)+k):
    seq_y.append( [ [ele] for ele in y[i-k:i+k+1] ] )

x_train, x_test, y_train, y_test = train_test_split(
    data.X, seq_y, test_size=0.3, shuffle=False
)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

aux_data = Tdiff(path, uniform=False) + MutualInfo(path, uniform=False)
aux_data.union()

aux_data.X = [ [ele['t-diff'], ele['mi-info']] for ele in aux_data.X ]

_x_train, _x_test, _y_train, _y_test = train_test_split(
    aux_data.X, aux_data.Y, test_size=0.3, shuffle=False
)
_x_train = np.array(_x_train)
_x_test = np.array(_x_test)

print('start training')
inputs = Input(shape=(len(x_test[0]), len(x_test[0][0])))
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inputs)
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
lstm_output = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)

aux_input = Input(shape=(len(_x_test[0]),))
x = Dense(50, activation='relu')(aux_input)
x = Dense(50, activation='relu')(x)
x = Dense(50, activation='relu')(x)
x = Dense(50, activation='relu')(x)
x = RepeatVector(len(x_test[0]))(x)
aux_output = TimeDistributed(Dense(len(x_test[0][0]), activation='relu'))(x)

x = concatenate([lstm_output, aux_output])
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
main_output = TimeDistributed(Dense(1, activation='sigmoid'))(x)

model = Model(inputs=[inputs, aux_input], outputs=main_output)
keras.utils.plot_model(model, to_file='model.png')
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
ts = TensorBoard(log_dir='./log')
model.fit([x_train, _x_train], y_train, batch_size=100, callbacks=[early_stop, ts], validation_split=0.1, epochs=100)
pred = model.predict([x_test, _x_test])
choose = lambda x : [ ele[k] for ele in x ]
y_pred = data.y2lab(choose(pred))
y_test = data.y2lab(choose(y_test))
print(metrics.flat_classification_report(
    y_test, y_pred, labels=('I', 'E'), digits=4
))
