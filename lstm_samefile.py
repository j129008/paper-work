import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from lib.lstmlib import *

path = './data/data_shuffle.txt'
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')

k = 10
x_train, x_test, y_train, y_test = context_data(path, k)
model = basic_model(x_test)
model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=0.01, epochs=100)
pred = model.predict([x_test])
print('same file')
score = report(pred, y_test)

