import pickle
from lib.feature import *
from keras.models import load_model
from sklearn_crfsuite import metrics
from lib.lstmlib import *
from lib.metric import pred2text, text2score
from lib.data import Data
import numpy as np
import re
from argparse import ArgumentParser

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

parser = ArgumentParser()
parser.add_argument('-i', dest='input', help='input file path')
parser.add_argument('-o', dest='output', help='output file path')
parser.add_argument('-ans', dest='ans', help='ans file path')
parser.add_argument('-voc', dest='voc', help='vocabulary list path')
parser.add_argument('-office', dest='office', default=None, help='office file path')
args = parser.parse_args()

text = open(args.input, 'r').read()
f = open(args.output, 'w')
refix = refix(args.voc, text)
if args.office != None:
    refix = office_fix(args.office, refix)
f.write(refix)
f.close()

print(text2score(args.ans, args.output))
