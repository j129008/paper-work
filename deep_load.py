import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from lib.feature import *
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, concatenate
from sklearn_crfsuite import metrics
import keras
from keras.callbacks import EarlyStopping, TensorBoard, CSVLogger
from keras.optimizers import Adam
from keras.layers import CuDNNLSTM, TimeDistributed, SimpleRNN, Embedding, RNN, GRU, Bidirectional, CuDNNLSTM
from sklearn.model_selection import train_test_split
from keras.models import load_model
from glob import glob


def pred2text(text_path, pred):
    i = 0
    output = ''
    for line in open(text_path):
        _line = line.strip()
        for char in _line:
            if y_pred[i] == 'E':
                output += char + '，'
            else:
                output += char
            i += 1
        output += '\n'
    print(output, file=open(text_path + '.res', 'w'))

k = 12
model = load_model('./pickles/keras.h5')

for f_name in glob('./data/budd_raw/proc_*'):
    new_data = VecContext(f_name, k=k, vec_size=100, w2v_text='./data/budd_w2v.txt')
    X = np.array(new_data.X)
    pred = model.predict([X])
    y_pred = new_data.y2lab(pred, threshold=0.5)
    pred2text(f_name, y_pred)
