import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from lib.lstmlib import *

train_path = './data/train.txt'
test_path = './data/test.txt'
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
k = 10

deep_train = VecContext(train_path, k=k, vec_size=50, w2v_text='./data/w2v.txt')
deep_test = VecContext(test_path, k=k, vec_size=50, w2v_text='./data/w2v.txt')
model = basic_model(deep_train.X)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
model.fit([deep_train.X], deep_train.Y, batch_size=100, callbacks=[early_stop], validation_split=0.01, epochs=100)
deep_pred = model.predict([deep_test.X])
print('diff file')
score = report(deep_pred, deep_test.Y)
