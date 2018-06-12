import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import pickle
from lib.feature import *
from keras.models import load_model
from sklearn_crfsuite import metrics
from lib.lstmlib import *
import numpy as np
from lib.arg import lstm_crf_arg

parser = lstm_crf_arg()
args = parser.parse_args()
test_path  = args.input
w2v_text = args.w2v
vec = args.vec

crf_k      = args.ck
deep_k     = args.lk

# CRF
crf_test  = Context(test_path, k=crf_k)
crf = pickle.load(open(args.cmod, 'rb'))
crf_pred = crf.predict_prob(crf_test.X)
union = lambda x: [ ins for chap in x for ins in chap ]
crf_pred = [ ele['E'] for ele in union(crf_pred) ]
ans = union(crf_test.Y)

# LSTM
deep_test = VecContext(test_path, k=deep_k, vec_size=50, w2v_text='../data/w2v.txt')
model = load_model(args.lmod)
deep_pred = model.predict([deep_test.X])

# ensemble
avg = lambda x, y: [ (x[i]+y[i])/2 for i in range(len(x)) ]
crf_deep = avg(crf_pred, deep_pred)
label_crf_deep = VecContext.y2lab(crf_deep)
label_deep = VecContext.y2lab(deep_pred)
label_crf = VecContext.y2lab(crf_pred)

print('average:')
print(metrics.flat_classification_report(
    ans, label_crf_deep, labels=('I', 'E'), digits=4
))
print('LSTM:')
print(metrics.flat_classification_report(
    ans, label_deep, labels=('I', 'E'), digits=4
))
print('CRF:')
print(metrics.flat_classification_report(
    ans, label_crf, labels=('I', 'E'), digits=4
))
