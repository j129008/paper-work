import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from lib.lstmlib import *

path = '../data/data_lite.txt'
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
k_baseline = 10

# seq2seq
print('s2s')
x_train, x_test, y_train, y_test = context_data(path, k=k_baseline, seq=True)
model = basic_model(x_test, seq=True)
model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.1, epochs=100)
pred = model.predict([x_test])
choose = lambda x : [ ele[k_baseline] for ele in x ]
score = report(choose(pred), choose(y_test))

# seq2seq + encoder/decoder
print('encoder')
x_train, x_test, y_train, y_test = context_data(path, k=k_baseline, seq=True)
model = encoder_model(x_test)
model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.1, epochs=100)
pred = model.predict([x_test])
score = report(choose(pred), choose(y_test))
