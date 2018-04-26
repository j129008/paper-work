import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import pickle
from lib.feature import *
from keras.models import load_model
from sklearn_crfsuite import metrics
import lightgbm as lgb
import numpy as np

test_path = './data/test.txt'
crf_k     = 5
deep_k    = 12
lgb_k = 4

# CRF
crf_test  = Context(test_path, k=crf_k)
crf = pickle.load(open('./pickles/crf_model.pkl', 'rb'))
crf_pred = crf.predict_prob(crf_test.X)
union = lambda x: [ ins for chap in x for ins in chap ]
crf_pred = [ ele['E'] for ele in union(crf_pred) ]
ans = union(crf_test.Y)

# LSTM
deep_test = VecContext(test_path, k=deep_k, vec_size=50, w2v_text='./data/w2v.txt')
model = load_model('./pickles/keras.h5')
deep_pred = model.predict([np.array(deep_test.X)])
deep_pred = union(deep_pred)

# lgb
lgb_test = UniVec(test_path, k=lgb_k, vec_size=50)
bst = lgb.Booster(model_file='./pickles/lgb.md')
bst_pred = bst.predict(lgb_test.X)

# ensemble
avg = lambda x, y, z: [ (x[i]+y[i]+z[i])/3 for i in range(len(x)) ]
crf_deep = avg(crf_pred, deep_pred, bst_pred)
label_crf_deep = deep_test.y2lab(crf_deep)
label_deep = deep_test.y2lab(deep_pred)
label_crf = deep_test.y2lab(crf_pred)
label_lgb = deep_test.y2lab(bst_pred)

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
print('LGB:')
print(metrics.flat_classification_report(
    ans, label_lgb, labels=('I', 'E'), digits=4
))
