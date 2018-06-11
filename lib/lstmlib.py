from lib.feature import *
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
import csv
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate, RepeatVector
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import CuDNNLSTM, TimeDistributed, Bidirectional
from keras.utils import plot_model

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

def context_data(path, k=10, size=None, seq=False):
    data = VecContext(path, k=k, vec_size=50)
    if seq == True:
        seq_y = []
        y = [0]*k + list(data.Y) + [0]*k
        for i in range(k, len(data.Y)+k):
            seq_y.append( [ [ele] for ele in y[i-k:i+k+1] ] )
        data.Y = seq_y

    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.Y, test_size=0.3, shuffle=False
    )

    if size != None:
        size_m = int(len(x_train)*size)
        x_train = x_train[:size_m]
        y_train = y_train[:size_m]
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test

def aux_data(path):
    def lab2val(l):
        if l[0] == 'O':
            return 1
        elif l[0] == 'B':
            return 2
        elif l[0] == 'I':
            return 3
        elif l[0] == 'E':
            return 4
        return 0
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

def aux_adv_data(path):
    def lab2val(l, tag):
        if l[0] == 'O':
            return 1 + tag*10
        elif l[0] == 'B':
            return 2 + tag*10
        elif l[0] == 'I':
            return 3 + tag*10
        elif l[0] == 'E':
            return 4 + tag*10
        return 0
    office = Label(path, lab_name='office', lab_file='./ref/tang_name/tangOffice.clliu.txt')
    nianhao = Label(path, lab_name='nianhao', lab_file='./ref/tang_name/tangReignperiods.clliu.txt')
    address = Label(path, lab_name='address', lab_file='./ref/tang_name/tangAddresses.clliu.txt')
    tdiff = Tdiff(path, uniform=False, noise=True)
    pmi = MutualInfo(path, uniform=False, noise=True)
    aux_data = office + nianhao + address + pmi + tdiff
    aux_data.union()
    aux_data.X = [ [ele['t-diff'], ele['mi-info'], lab2val(ele['office'], 1), lab2val(ele['address'], 2), lab2val(ele['nianhao'], 3)] for ele in aux_data.X ]
    x_train, x_test, y_train, y_test = train_test_split(
        aux_data.X, aux_data.Y, test_size=0.3, shuffle=False
    )
    return x_train, x_test

def basic_model(data, stack=5, seq=False):
    inputs = Input(shape=(len(data[0]), len(data[0][0])))
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inputs)
    for _ in range(stack-2):
        x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    if seq == True:
        x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
        main_output = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    else:
        x = Bidirectional(CuDNNLSTM(50))(x)
        main_output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inputs], outputs=main_output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def encoder_model(data, n_encoder, n_decoder, n_lstm):
    inputs = Input(shape=(len(data[0]), len(data[0][0])))
    if n_lstm != 0:
        # encoder
        x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inputs)
        for _ in range(n_encoder-2):
            x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(50, return_sequences=False))(x)
        coder_output = RepeatVector(len(data[0]))(x)

        # stack lstm
        x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inputs)
        for _ in range(n_lstm-2):
            x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
        lstm_output = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
        x = concatenate([lstm_output, coder_output])

        # decoder
        x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
        for _ in range(n_decoder-1):
            x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    else:
        # encoder
        x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inputs)
        for _ in range(n_encoder-2):
            x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(50, return_sequences=False))(x)
        x = RepeatVector(len(data[0]))(x)

        # decoder
        x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
        for _ in range(n_decoder-1):
            x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
    main_output = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    model = Model(inputs=[inputs], outputs=main_output)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
