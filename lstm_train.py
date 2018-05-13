import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from lib.feature import *
from keras.models import load_model
import numpy as np
from lib.lstmlib import *

train_path = './data/train-valid.txt'
valid_path = './data/valid.txt'
deep_k     = 10

deep_valid = VecContext(valid_path, k=deep_k, vec_size=50, w2v_text='./data/w2v.txt')
deep_train = VecContext(train_path, k=deep_k, vec_size=50, w2v_text='./data/w2v.txt')
model = basic_model(deep_train.X)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
model.fit([deep_train.X], deep_train.Y, batch_size=100, callbacks=[early_stop], validation_data=(np.array(deep_valid.X), np.array(deep_valid.Y)), epochs=100)
model.save('./pickles/lstm_train-valid.h5')
