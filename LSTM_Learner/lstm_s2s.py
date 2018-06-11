import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from lib.lstmlib import *
from lib.arg import lstm_arg

parser = lstm_arg()
parser.add_argument('--encoder', '-enc', dest='enc', type=int, default=5)
parser.add_argument('--decoder', '-dec', dest='dec', type=int, default=5)
parser.add_argument('--stack', '-st', dest='stack', type=int, default=5)
args = parser.parse_args()
path = args.input
k = args.k
w2v_text = args.w2v
vec = args.vec
stack = args.stack
trainsplit = args.trainsplit
patience = args.patience
valid = args.valid
enc = args.enc
dec = args.dec
stack = args.stack

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, mode='min')
k_baseline = k

if enc != True:
    # seq2seq
    x_train, x_test, y_train, y_test = context_data(path, k=k_baseline, seq=True, train_size=trainsplit)
    model = basic_model(x_test, seq=True, stack=stack)
    model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=valid, epochs=100)
    pred = model.predict([x_test])
    choose = lambda x : [ ele[k_baseline] for ele in x ]
    score = report(choose(pred), choose(y_test))
    if args.smod != None:
        model.save(args.smod)
else:
    # seq2seq + encoder/decoder + lstm stack
    x_train, x_test, y_train, y_test = context_data(path, k=k_baseline, seq=True)
    model = encoder_model(x_test, n_decoder=dec, n_encoder=enc, n_lstm=stack)
    model.fit([x_train], y_train, batch_size=100, callbacks=[early_stop], validation_split=valid, epochs=100)
    pred = model.predict([x_test])
    choose = lambda x : [ ele[k_baseline] for ele in x ]
    score = report(choose(pred), choose(y_test))
    if args.smod != None:
        model.save(args.smod)
