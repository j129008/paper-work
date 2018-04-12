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
k = 9
data = VecContext(path, k=k, vec_size=50)
x_train, x_test, y_train, y_test = train_test_split(
    data.X, data.Y, test_size=0.3, shuffle=False
)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

try:
    office = pickle.load(open('./pickles/office.pkl', 'rb'))
    nianhao = pickle.load(open('./pickles/nianhao.pkl', 'rb'))
    entry = pickle.load(open('./pickles/entry.pkl', 'rb'))
    address = pickle.load(open('./pickles/address.pkl', 'rb'))
    name = pickle.load(open('./pickles/name.pkl', 'rb'))
except:
    office = Label(path, lab_name='office', lab_file='./ref/known/office2.txt')
    nianhao = Label(path, lab_name='nianhao', lab_file='./ref/known/nianhao.txt')
    entry = Label(path, lab_name='entry', lab_file='./ref/known/entry1.txt')
    address = Label(path, lab_name='address', lab_file='./ref/known/address2.txt')
    name = Label(path, lab_name='name', lab_file='./ref/known/name5.txt')
    office.save('./pickles/office.pkl')
    nianhao.save('./pickles/nianhao.pkl')
    entry.save('./pickles/entry.pkl')
    address.save('./pickles/address.pkl')
    name.save('./pickles/name.pkl')

aux_data = Tdiff(path, uniform=False) + MutualInfo(path, uniform=False) + office + nianhao + entry + address + name
aux_data.union()
def lab2val(l):
    if l[0] == 'O':
        return 0
    elif l[0] == 'B':
        return 1
    else:
        return 2
aux_data.X = [ [ele['t-diff'], ele['mi-info'], lab2val(ele['office']), lab2val(ele['name']), lab2val(ele['address']), lab2val(ele['nianhao']), lab2val(ele['entry'])] for ele in aux_data.X ]
_x_train, _x_test, _y_train, _y_test = train_test_split(
    aux_data.X, aux_data.Y, test_size=0.3, shuffle=False
)
_x_train = np.array(_x_train)
_x_test = np.array(_x_test)

inputs = Input(shape=(len(x_test[0]), len(x_test[0][0])))
x = Bidirectional(CuDNNLSTM(50, return_sequences=True, go_backwards=True))(inputs)
for _ in range(4):
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True, go_backwards=True))(x)
lstm_output= Bidirectional(CuDNNLSTM(50))(x)

aux_input = Input(shape=(len(_x_test[0]),))
x = concatenate([lstm_output, aux_input])
x = Dense(50, activation='relu')(x)
x = Dense(50, activation='relu')(x)
x = Dense(50, activation='relu')(x)
main_output = Dense(1, activation='sigmoid')(x)

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
y_pred = data.y2lab(pred)
y_test = data.y2lab(y_test)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=('I', 'E'), digits=4
))
