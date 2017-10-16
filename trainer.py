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
        all_possible_transitions = True
    )

    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05)
    }

    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=['I', 'E'])

    rs = RandomizedSearchCV(crf, params_space,
                            cv      = 3,
                            verbose = 1,
                            n_jobs  = 1,
                            n_iter  = 50,
                            scoring = f1_scorer)

    rs.fit([ [x] for x in X ], Y)

    return rs

rs = randomCV()
print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
pickle.dump(rs.best_estimator_, open('./pickles/crf_best_model' + '.pkl', 'wb'))
