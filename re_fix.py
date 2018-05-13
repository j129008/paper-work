import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import pickle
from lib.feature import *
from keras.models import load_model
from sklearn_crfsuite import metrics
from lib.lstmlib import *
from lib.metric import pred2text
from lib.data import Data
import numpy as np
import re

def refix(ne_path, text):
    ne_list = lambda path:[ line.strip() for line in open(path, 'r') ]
    err_list = lambda word: [ word[:i] + '，' + word[i:] for i in range(1, len(word)) ]
    _text = text[:]
    for ne in ne_list(ne_path):
        for err in err_list(ne):
            _text = _text.replace(err, ne)
    return _text

def office_fix(office_path, text):
    ne_list = lambda path:[ line.strip() for line in open(path, 'r') ]
    _text = text[:]
    for ne in ne_list(office_path):
        _text = _text.replace(ne, '<'+ne+'>')
    _text = _text.replace('>，<', '')
    _text = _text.replace('>', '')
    _text = _text.replace('<', '')
    return _text

def text2score(ans_path, pred_path):
    ans_data = Data(ans_path)
    pred_data = Data(pred_path)
    print(metrics.flat_classification_report(
        ans_data.Y, pred_data.Y, labels=('I', 'E'), digits=4
    ))

test_path = './data/test.txt'
crf_k     = 5
deep_k    = 10

# LSTM
deep_test = VecContext(test_path, k=deep_k, vec_size=50, w2v_text='./data/w2v.txt')
model = load_model('./pickles/lstm-train.h5')
deep_pred = model.predict([deep_test.X])
deep_pred_lab = VecContext.y2lab(deep_pred)

# CRF
crf_test  = Context(test_path, k=crf_k)
crf = pickle.load(open('./pickles/crf_model.pkl', 'rb'))
crf_pred = crf.predict(crf_test.X)
union = lambda x: [ ins for chap in x for ins in chap ]
crf_pred = union(crf_pred)
ans = crf_test.Y

print('crf predict')
print(metrics.flat_classification_report(
    ans, crf_pred, labels=('I', 'E'), digits=4
))

print('lstm predict')
print(metrics.flat_classification_report(
    ans, deep_pred_lab, labels=('I', 'E'), digits=4
))

print('re fix lstm predict')
lstm_pred_text = pred2text(test_path, deep_pred_lab)
lstm_pred_text = refix('./ref/tang_name/tangOffice.clliu.txt', lstm_pred_text)
lstm_pred_text = refix('./ref/tang_name/tangAddresses.clliu.txt', lstm_pred_text)
lstm_pred_text = refix('./ref/tang_name/tangReignperiods.clliu.txt', lstm_pred_text)
lstm_pred_text = office_fix(test_path, lstm_pred_text)
lstm_pred_path = './data/lstm_fix.txt'
f = open(lstm_pred_path, 'w')
f.write(lstm_pred_text)
text2score(test_path, lstm_pred_path)

print('re fix crf predict')
crf_pred_text = pred2text(test_path, crf_pred)
crf_pred_text = refix('./ref/tang_name/tangOffice.clliu.txt', crf_pred_text)
crf_pred_text = refix('./ref/tang_name/tangAddresses.clliu.txt', crf_pred_text)
crf_pred_text = refix('./ref/tang_name/tangReignperiods.clliu.txt', crf_pred_text)
crf_pred_text = office_fix(test_path, crf_pred_text)
crf_pred_path = './data/crf_fix.txt'
f = open(crf_pred_path, 'w')
f.write(crf_pred_text)
text2score(test_path, crf_pred_path)
