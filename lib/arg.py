import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from lib.feature import *
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

def crf_arg():
    parser = ArgumentParser()
    # basic option
    parser.add_argument('-i', dest='input', help='input file path')
    parser.add_argument('-smod', dest='save', default=None, help='model save path')
    parser.add_argument('-k', dest='k', default=1, type=int, help='context size k')
    parser.add_argument('-ts', dest='trainsplit', default=0.7, type=float, help='train test split size')
    parser.add_argument('-subtrain', dest='subtrain', type=float, default=1.0, help='set training data size')
    # feature option
    parser.add_argument('-rhy', dest='rhy', default=None, help='反切,聲母,韻目,調,等,呼,韻母')
    parser.add_argument('-list', dest='list', default=None, help='vocabulary list file path')
    parser.add_argument('-ngram', dest='ngram', default=2, type=int, help='set ngram size')
    parser.add_argument('-pmi', action='store_true', help='add feature pmi')
    parser.add_argument('-tdiff', action='store_true', help='add feature tdiff')
    return parser

def lstm_arg():
    parser = ArgumentParser()
    # basic option
    parser.add_argument('-i', dest='input', help='input file path')
    parser.add_argument('-smod', dest='smod', default=None, help='model save path')
    parser.add_argument('-k', dest='k', default=1, type=int, help='context size k')
    parser.add_argument('-ts', dest='trainsplit', default=0.7, type=float, help='train test split size')
    parser.add_argument('-subtrain', dest='subtrain', type=float, default=1.0, help='set training data size')
    # LSTM option
    parser.add_argument('-w2v', dest='w2v', default='./data/w2v.txt', help='w2v text path')
    parser.add_argument('-vec', dest='vec', default=50, type=int, help='w2v vec size')
    parser.add_argument('-val', dest='valid', type=float, default=0.1, help='valid data proportion in training data')
    parser.add_argument('-stack', dest='stack', default=5, type=int, help='layer stack size')
    parser.add_argument('-plot', dest='plot', default=None, help='path of model graph')
    parser.add_argument('-pat' , dest='patience', type=int, default=2, help='early stop patience')
    return parser

def lstm_crf_arg():
    parser = ArgumentParser()
    parser.add_argument('-train', dest='train', help='train file path')
    parser.add_argument('-test', dest='test', help='test file path')
    parser.add_argument('-pred', dest='pred', default=None, help='predict file path')
    parser.add_argument('-lk', dest='lk', default=10, type=int, help='lstm context k')
    parser.add_argument('-ck', dest='ck', default=5, type=int, help='crf context k')
    parser.add_argument('-lmod', dest='lmod', help='load lstm model')
    parser.add_argument('-cmod', dest='cmod', help='load crf model')
    parser.add_argument('-cpmi', action='store_true', help='load crf pmi')
    parser.add_argument('-ctdiff', action='store_true', help='load crf tdiff')
    parser.add_argument('-w2v', dest='w2v', default='./data/w2v.txt', help='w2v text path')
    parser.add_argument('-vec', dest='vec', default=50, type=int, help='w2v vec size')
    return parser

def crf_data(args):
    path = args.input
    data = Context(path, k=args.k, n_gram=args.ngram)
    if args.pmi:
        data += MutualInfo(path)
    if args.tdiff:
        data += Tdiff(path)
    if args.rhy != None:
        data += Rhyme(path, '../data/rhyme.txt', '../pickles/rhyme_list.pkl', args.rhy.split(','))
    if args.list != None:
        for name_path in args.list.split(','):
            lab_name, lab_file = name_path.split(':')
            data += Label(path, lab_name=lab_name, lab_file=lab_file)
    return data

def lstm_data(args):
    path = args.input
    if args.pmi:
        try:
            data += MutualInfo(path, uniform=False)
        except:
            data = MutualInfo(path, uniform=False)
    if args.tdiff:
        try:
            data += Tdiff(path, uniform=False)
        except:
            data = Tdiff(path, uniform=False)
    if args.rhy != None:
        try:
            data += VecRhyme(path, '../data/rhyme.txt', '../pickles/rhyme_list.pkl', args.rhy.split(','))
        except:
            data = VecRhyme(path, '../data/rhyme.txt', '../pickles/rhyme_list.pkl', args.rhy.split(','))
    if args.list != None:
        for name_path in args.list.split(','):
            lab_name, lab_file = name_path.split(':')
            try:
                data += VecLabel(path, lab_name=lab_name, lab_file=lab_file)
            except:
                data = VecLabel(path, lab_name=lab_name, lab_file=lab_file)
    data.union()
    if args.noise == True:
        data.shuffle()
    keys = [ key for key in data.X[0] ]
    if args.rhy != None:
        keys.remove(args.rhy)
        data.X = [ [ ins[k] for k in keys ]+ins[args.rhy] for ins in data.X ]
        keys.append(args.rhy)
    else:
        data.X = [ [ ins[k] for k in keys ] for ins in data.X ]
    x_train, x_test, y_train, y_test = train_test_split(
        data.X, data.Y, test_size=1.0-args.trainsplit, shuffle=False
    )
    return keys, x_train, x_test, y_train, y_test
