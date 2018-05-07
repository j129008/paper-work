import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from lib.feature import *
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
import csv
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import CuDNNLSTM, TimeDistributed, Bidirectional

def report(pred, truth):
    _pred = VecContext.y2lab(pred)
    _test = VecContext.y2lab(truth)
    print(metrics.flat_classification_report(
        _test, _pred, labels=('I', 'E'), digits=4
    ))
    label = 'E'
    P = metrics.flat_precision_score(_test, _pred, pos_label=label)
    R = metrics.flat_recall_score(_test, _pred, pos_label=label)
    f1 = metrics.flat_f1_score(_test, _pred, pos_label=label)
    return {'P':P, 'R':R, 'f1':f1}

def context_data(path, k=10, size=None):
    data = VecContext(path, k=k, vec_size=50)
    if size != None:
        size_m = int(len(data.X)*size)
        data.X = data.X[:size_m]
        data.Y = data.Y[:size_m]
    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.Y, test_size=0.3, shuffle=False
    )
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test

def aux_data():
    def lab2val(l):
        if l[0] == 'O':
            return 1
        elif l[0] == 'B':
            return 0
        elif l[0] == 'I':
            return 0
        elif l[0] == 'E':
            return 2
    office = Label(path, lab_name='office', lab_file='./ref/tang_name/tangOffice.clliu.txt')
    nianhao = Label(path, lab_name='nianhao', lab_file='./ref/tang_name/tangReignperiods.clliu.txt')
    address = Label(path, lab_name='address', lab_file='./ref/tang_name/tangAddresses.clliu.txt')
    aux_data = Tdiff(path, uniform=False) + MutualInfo(path, uniform=False) + office + nianhao + address
    aux_data.union()
    aux_data.X = [ [ele['t-diff'], ele['mi-info'], lab2val(ele['office']), lab2val(ele['address']), lab2val(ele['nianhao'])] for ele in aux_data.X ]
    x_train, x_test, y_train, y_test = train_test_split(
        aux_data.X, aux_data.Y, test_size=0.3, shuffle=False
    )
    return x_train, x_test

def basic_model(data, stack=5):
    inputs = Input(shape=(len(data[0]), len(data[0][0])))
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inputs)
    for _ in range(stack-2):
        x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    x= Bidirectional(CuDNNLSTM(50))(x)
    main_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inputs], outputs=main_output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

path = './data/data_lite.txt'
result_table = csv.writer( open('./csv/lstm_report.csv', 'w') )
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
k_baseline = 10

# context
result_table.writerow(['context k'])
for k in range(1, 2):
    x_train, x_test, y_train, y_test = context_data(path, k)
    model = basic_model(x_test)
    model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.01, epochs=100)
    pred = model.predict([x_test])
    score = report(pred, y_test)
    result_table.writerow([k, score['P'], score['R'], score['f1']])

# data size
result_table.writerow(['data size'])
for size in [0.1*i for i in range(1, 11)]:
    x_train, x_test, y_train, y_test = context_data(path, k=k_baseline, size)
    model = basic_model(x_test)
    model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.01, epochs=100)
    pred = model.predict([x_test])
    score = report(pred, y_test)
    result_table.writerow([size, score['P'], score['R'], score['f1']])

# stack size
result_table.writerow(['stack size'])
for stack in range(3, 10):
    x_train, x_test, y_train, y_test = context_data(path, k=k_baseline)
    model = basic_model(x_test, stack=stack)
    model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.01, epochs=100)
    pred = model.predict([x_test])
    score = report(pred, y_test)
    result_table.writerow([stack, score['P'], score['R'], score['f1']])

# features
x_train, x_test, y_train, y_test = context_data(path, k=k_baseline)
aux_train, aux_test = aux_data()
tdiff_train, mi_train, office_train, address_train, nianhao_train = list(zip(*aux_train))
tdiff_test, mi_test, office_test, address_test, nianhao_test = list(zip(*aux_test))

for aux_name, aux_train, aux_test in [ ('tdiff', tdiff_train, tdiff_test), ('pmi', mi_train, mi_test), ('office', office_train, office_test), ('address', address_train, address_test), ('nianhao', nianhao_train, nianhao_test) ]:
    aux_train = np.array(aux_train).reshape(-1, 1)
    aux_test = np.array(aux_test).reshape(-1, 1)
    inputs = Input(shape=(len(x_test[0]), len(x_test[0][0])))
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inputs)
    for _ in range(3):
        x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    lstm_output= Bidirectional(CuDNNLSTM(50))(x)

    aux_input = Input(shape=(len(aux_train[0]),))
    x = concatenate([lstm_output, aux_input])
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    main_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inputs, aux_input], outputs=main_output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit([x_train, aux_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.1, epochs=100)
    pred = model.predict([x_test, aux_test])
    score = report(pred, y_test)
    result_table.writerow([aux_name, score['P'], score['R'], score['f1']])
