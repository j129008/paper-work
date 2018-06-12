import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from lib.lstmlib import *
from lib.arg import lstm_arg, lstm_data

parser = lstm_arg()
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

# context
x_train, x_test, y_train, y_test = context_data(path, k=k, vec_size=vec, w2v_text=w2v_text)
model = basic_model(x_test, stack=stack)
model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=valid, epochs=100)
pred = model.predict([x_test])
print('context k =', k)
score = report(pred, y_test)
if args.smod != None:
    model.save(args.smod)
