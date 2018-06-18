import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from lib.lstmlib import *
from lib.arg import lstm_arg, lstm_data
from lib.metric import pred2text
from keras.models import load_model


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
subtrain=args.subtrain

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, mode='min')

# context
x_train, x_test, y_train, y_test = context_data(path, k=k, vec_size=vec, w2v_text=w2v_text, train_size=0.0, size=subtrain)
model = load_model(args.smod)
pred = model.predict([x_test])
pred = [ 'E' if ele > 0.5 else 'I' for ele in pred ]
print(pred2text(args.input, pred))
