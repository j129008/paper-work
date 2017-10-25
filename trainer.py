from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from tqdm import tqdm as bar
from lib.death_book import *
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import random

books = [ book( txt ) for txt in bar ( txt_loader('./data/all_rmBr.txt') ) ]
random.shuffle(books)

X, Y = [], []
for book in bar( books ):
    X.extend( [ feature for feature in book.feature_list ] )
    Y.extend( [ end_lab for end_lab in book.end_label ] )

def loadLabel():
    print('load label txt')
    txt_no_comma_all = ''
    for book in bar( books ):
        txt_no_comma_all += book.no_comma_text
    lab_addr = label(txt_no_comma_all, txt_loader('./ref/known/address2.txt'), 'address')
    lab_office = label(txt_no_comma_all, txt_loader('./ref/known/office2.txt'), 'office')
    lab_name = label(txt_no_comma_all, txt_loader('./ref/known/name5.txt'), 'name')
    lab_nianhao = label(txt_no_comma_all, txt_loader('./ref/known/nianhao.txt'), 'nianhao')
    lab_entry = label(txt_no_comma_all, txt_loader('./ref/known/entry1.txt'), 'entry')

    print('insert label into feature')
    for i in range(len(X)):
        X[i]['addr'] = lab_addr[i]
        X[i]['office'] = lab_office[i]
        X[i]['name'] = lab_name[i]
        X[i]['nianhao'] = lab_nianhao[i]
        X[i]['entry'] = lab_entry[i]

loadLabel()

pickle.dump(X, open('./pickles/X_all.pkl', 'wb'))
pickle.dump(Y, open('./pickles/Y_all.pkl', 'wb'))

X, X_private, Y, Y_private = train_test_split(
    X, Y, test_size=0.4, shuffle=False
)

pickle.dump(X, open('./pickles/X.pkl', 'wb'))
pickle.dump(Y, open('./pickles/Y.pkl', 'wb'))
pickle.dump(X_private, open('./pickles/X_private.pkl', 'wb'))
pickle.dump(Y_private, open('./pickles/Y_private.pkl', 'wb'))

X = np.array(X)
Y = np.array(Y)

def randomCV():
    crf = sklearn_crfsuite.CRF(
        algorithm                = 'lbfgs',
        max_iterations           = 100,
        all_possible_transitions = True,
        c1 = 0.01650478417296183,
        c2 = 0.17925029793689362
    )

    params_space = {
        #  'c1': scipy.stats.expon(scale=0.5),
        #  'c2': scipy.stats.expon(scale=0.05)
    }

    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=['I', 'E'])

    rs = RandomizedSearchCV(crf, params_space,
                            cv      = 3,
                            verbose = 1,
                            n_jobs  = 8,
                            n_iter  = 1,
                            scoring = f1_scorer)

    rs.fit([ [x] for x in X ], Y)

    return rs

rs = randomCV()
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
pickle.dump(rs.best_estimator_, open('./pickles/crf_best_model' + '.pkl', 'wb'))
