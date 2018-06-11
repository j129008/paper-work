import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import pickle
from lib.feature import *
from keras.models import load_model
from sklearn_crfsuite import metrics
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
from itertools import chain
from lib.lstmlib import *

test_path  = '../data/test_lite.txt'
valid_path = '../data/valid_lite.txt'
crf_k      = 5
deep_k     = 10

# CRF
crf_test  = Context(test_path, k=crf_k)
crf = pickle.load(open('./pickles/crf_model-best.pkl', 'rb'))
crf_pred = crf.predict_prob(crf_test.X)
union = lambda x: [ ins for chap in x for ins in chap ]
crf_pred = [ ele['E'] for ele in union(crf_pred) ]

# CRF valid
crf_valid  = Context(valid_path, k=crf_k)
crf_valid_pred = crf.predict_prob(crf_valid.X)
crf_valid_pred = [ ele['E'] for ele in union(crf_valid_pred) ]

# set answer
ans = union(crf_test.Y)

# LSTM
deep_test = VecContext(test_path, k=deep_k, vec_size=50, w2v_text='./data/w2v.txt')
deep_valid = VecContext(valid_path, k=deep_k, vec_size=50, w2v_text='./data/w2v.txt')
model = load_model('./pickles/lstm_train-valid.h5')
deep_pred = model.predict([deep_test.X])

# LSTM valid
deep_valid_pred = model.predict([np.array(deep_valid.X)])
deep_valid_pred = union(deep_valid_pred)

## rf valid train
concat = lambda x,y : [ [ele[0], ele[1]] for ele in zip(x,y) ]
rf_valid = UniVec(valid_path, vec_size=50)
rf = svm.SVC(C=1000)
rf.fit( concat(crf_valid_pred, deep_valid_pred) , rf_valid.Y )

# ensemble
rf_valid = UniVec(test_path, vec_size=50)
ensemble_pred = rf.predict(concat(crf_pred, deep_pred))
label_deep = VecContext.y2lab(deep_pred)
label_crf = VecContext.y2lab(crf_pred)
label_ensemble = VecContext.y2lab(ensemble_pred)

print('ensemble:')
print(metrics.flat_classification_report(
    ans, label_ensemble, labels=('I', 'E'), digits=4
))
print('LSTM:')
print(metrics.flat_classification_report(
    ans, label_deep, labels=('I', 'E'), digits=4
))
print('CRF:')
print(metrics.flat_classification_report(
    ans, label_crf, labels=('I', 'E'), digits=4
))
