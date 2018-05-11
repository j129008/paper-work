import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from lib.lstmlib import *

path = './data/data_shuffle.txt'
result_table = csv.writer( open('./csv/lstm_report.csv', 'w') )
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
k_baseline = 10

# context
result_table.writerow(['context k'])
for k in range(1, 11):
    x_train, x_test, y_train, y_test = context_data(path, k)
    model = basic_model(x_test)
    model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.01, epochs=100)
    pred = model.predict([x_test])
    print('context k =', k)
    score = report(pred, y_test)
    result_table.writerow([k, score['P'], score['R'], score['f1']])

# data size
result_table.writerow(['data size'])
for size in [0.1*i for i in range(1, 11)]:
    x_train, x_test, y_train, y_test = context_data(path, k=k_baseline, size=size)
    model = basic_model(x_test)
    model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.01, epochs=100)
    pred = model.predict([x_test])
    print('data size:', size)
    score = report(pred, y_test)
    result_table.writerow([size, score['P'], score['R'], score['f1']])

# stack size
result_table.writerow(['stack size'])
for stack in range(3, 10):
    x_train, x_test, y_train, y_test = context_data(path, k=k_baseline)
    model = basic_model(x_test, stack=stack)
    model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.01, epochs=100)
    pred = model.predict([x_test])
    print('stack size:', stack)
    score = report(pred, y_test)
    result_table.writerow([stack, score['P'], score['R'], score['f1']])

# features
x_train, x_test, y_train, y_test = context_data(path, k=k_baseline)
aux_train, aux_test = aux_data(path)
tdiff_train, mi_train, office_train, address_train, nianhao_train = list(zip(*aux_train))
tdiff_test, mi_test, office_test, address_test, nianhao_test = list(zip(*aux_test))

for aux_name, aux_train, aux_test in [ ('tdiff', tdiff_train, tdiff_test), ('pmi', mi_train, mi_test), ('office', office_train, office_test), ('address', address_train, address_test), ('nianhao', nianhao_train, nianhao_test) ]:
    print(aux_name)
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

# seq2seq
print('s2s')
x_train, x_test, y_train, y_test = context_data(path, k=k_baseline, seq=True)
model = basic_model(x_test, seq=True)
model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.1, epochs=100)
pred = model.predict([x_test])
choose = lambda x : [ ele[k_baseline] for ele in x ]
score = report(choose(pred), choose(y_test))
result_table.writerow([aux_name, score['P'], score['R'], score['f1']])

# seq2seq + encoder/decoder
print('encoder')
x_train, x_test, y_train, y_test = context_data(path, k=k_baseline, seq=True)
model = encoder_model(x_test)
model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.1, epochs=100)
pred = model.predict([x_test])
score = report(choose(pred), choose(y_test))
result_table.writerow([aux_name, score['P'], score['R'], score['f1']])
