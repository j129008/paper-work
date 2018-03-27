from lib.feature import *
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from sklearn_crfsuite import metrics
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from keras.optimizers import Adam
from keras.layers import LSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU, Bidirectional, CuDNNLSTM
from sklearn.model_selection import train_test_split

path = './data/data_test.txt'
k = 5
data = VecContext(path, k=k)
x_train, x_test, y_train, y_test = train_test_split(
    data.X, data.Y, test_size=0.6, shuffle=False
)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

aux_data = Tdiff(path, uniform=False)
aux_data.union()
aux_data.X = [ ele['t-diff'] for ele in aux_data.X ]
_x_train, _x_test, _y_train, _y_test = train_test_split(
    aux_data.X, aux_data.Y, test_size=0.6, shuffle=False
)
_x_train = np.array(_x_train)
_x_test = np.array(_x_test)

inputs = Input(shape=(len(x_test[0]), len(x_test[0][0])))
x = Bidirectional(CuDNNLSTM(50, return_sequences=True, go_backwards=True))(inputs)
for _ in range(4):
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True, go_backwards=True))(x)
x = Bidirectional(CuDNNLSTM(50))(x)
lstm_output = Dense(1, activation='sigmoid')(x)

aux_input = Input(shape=(1,))
aux_output = Dense(1, activation='sigmoid')(lstm_output)
x = concatenate([lstm_output, aux_input])
main_output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[inputs, aux_input], outputs=[main_output, aux_output])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
ts = TensorBoard(log_dir='./log')
model.fit([x_train, _x_train], [y_train, y_train], batch_size=100, callbacks=[early_stop, ts], validation_split=0.1, epochs=100)
pred = model.predict([x_test, _x_test])
y_pred = data.y2lab(pred)
y_test = data.y2lab(y_test)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=('I', 'E'), digits=4
))
