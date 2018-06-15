import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from lib.lstmlib import *
from lib.arg import lstm_arg, lstm_data

parser = lstm_arg()
parser.add_argument('-pmi', action='store_true')
parser.add_argument('-tdiff', action='store_true')
parser.add_argument('-noise', action='store_true')
parser.add_argument('-rhy', dest='rhy', default=None, help='反切,聲母,韻目,調,等,呼,韻母')
parser.add_argument('-list', dest='list', default=None)
args = parser.parse_args()
path = args.input
k = args.k
w2v_text = args.w2v
vec = args.vec
stack = args.stack
trainsplit = args.trainsplit
patience = args.patience
valid = args.valid

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, mode='min')
k_baseline = k

# features
x_train, x_test, y_train, y_test = context_data(path, k=k_baseline, vec_size=vec, w2v_text=w2v_text, train_size=trainsplit)
aux_name, aux_x_train, aux_x_test, aux_y_train, aux_y_test = lstm_data(args)

print(aux_name)
aux_train = np.array(aux_x_train).reshape(-1, 1)
aux_test = np.array(aux_x_test).reshape(-1, 1)
inputs = Input(shape=(len(x_test[0]), len(x_test[0][0])))
x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(inputs)
for _ in range(stack-2):
    x = Bidirectional(CuDNNLSTM(50, return_sequences=True))(x)
lstm_output= Bidirectional(CuDNNLSTM(50))(x)

aux_input = Input(shape=(len(aux_x_train[0]),))
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

model.fit([x_train, np.array(aux_x_train)], y_train, batch_size=100, callbacks=[early_stop], validation_split=valid, epochs=100)
pred = model.predict([x_test, np.array(aux_x_test)])
score = report(pred, y_test)
