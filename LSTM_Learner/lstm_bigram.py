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
from random import randint
from sklearn.utils import shuffle
from lib.arg import lstm_arg

parser = lstm_arg()
args = parser.parse_args()
path = args.input
k = args.k
w2v_text = args.w2v
vec = args.vec
stack = args.stack
trainsplit = args.trainsplit

data = VecContext(path, k=k, vec_size=vec, w2v_text=w2v_text)
vec_adder = lambda v1, v2: [ v1[i]+v2[i] for i in range(len(v1)) ]
adder = lambda v_list : [ vec_adder(v_list[i], v_list[i+1]) for i in range(len(v_list)-1) ]
bigram_adder = lambda v_list : list(adder(v_list)) + list(v_list)
data_trans = lambda x : np.array([ bigram_adder(ele) for ele in x])

def batch_gen(x, y, batch_size=100):
    seg_cnt = len(y)//batch_size
    while True:
        for _ in range(seg_cnt):
            i = randint(0, seg_cnt)
            _x = x[i*batch_size:(i+1)*batch_size]
            batch_x = np.array([ bigram_adder(vec_list) for vec_list in _x ])
            batch_y = np.array(y[i*batch_size:(i+1)*batch_size])
            yield (batch_x, batch_y)

x_train, x_test, y_train, y_test = train_test_split(
    data.X, data.Y, test_size=1.0-trainsplit, shuffle=False
)
split_p = -len(x_train)//10
x_vaild = x_train[split_p:]
y_vaild = y_train[split_p:]
x_train = x_train[:split_p]
y_train = y_train[:split_p]

inputs = Input(shape=((2*k+1) + 2*k, vec))
x = Bidirectional(CuDNNLSTM(50, return_sequences=True, go_backwards=True))(inputs)
for _ in range(stack-2):
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True, go_backwards=True))(x)
x= Bidirectional(CuDNNLSTM(50))(x)
main_output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[inputs], outputs=main_output)
if args.plot != None:
    keras.utils.plot_model(model, to_file=args.plot)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='min')
model.fit_generator(batch_gen(x_train, y_train, batch_size=100), steps_per_epoch=len(y_train)//100, epochs=100, validation_data=(data_trans(x_vaild), y_vaild), callbacks=[early_stop])
pred = model.predict([data_trans(x_test)])
y_pred = VecContext.y2lab(pred)
y_test = VecContext.y2lab(y_test)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=('I', 'E'), digits=4
))
