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

path = './data/budd_proc.txt'
k = 12
new_data = VecContext('./data/budd_test.txt', k=k, vec_size=50, w2v_text='./data/budd_w2v.txt')
X = np.array(new_data.X)
model = load_model('./pickles/keras.h5')
pred = model.predict([X])
y_pred = new_data.y2lab(pred, threshold=0.5)
i = 0
output = ''
for line in open('./data/budd_test.txt'):
    _line = line.strip()
    for char in _line:
        if y_pred[i] == 'E':
            output += char + '，'
        else:
            output += char
        i += 1
    output += '\n'
print(output, file=open('budd_output.txt', 'w'))
