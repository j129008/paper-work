from lib.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn_crfsuite import metrics, CRF
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
from lib.crf import *
import logging
import numpy as np
import scipy
from collections import Counter
import pickle
import pdb

logging.basicConfig(level=logging.DEBUG)

class Learner(Data):
    def __init__(self, data, random_state=None, train_size=0.7):
        self.X = data.X
        self.Y = data.Y
        self.split_data(random_state=random_state, train_size=train_size)
    def split_data(self, train_size=0.7, random_state=None, shuffle=False):
        self.random_state = random_state
        self.X_train, self.X_private, self.Y_train, self.Y_private = train_test_split(
            self.X, self.Y, test_size=1.0-train_size, random_state=random_state, shuffle=shuffle
        )
    def get_CRF(self, c1=0, c2=1):
        crf = CRF(
            algorithm                = 'lbfgs',
            max_iterations           = 1000,
            all_possible_transitions = True,
            c1 = c1,
            c2 = c2
        )
        return crf
    def train_CV(self, c1=0, c2=1, cv=3, n_iter=1):
        crf = self.get_CRF(c1=c1, c2=c2)
        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted',
                                labels=['I', 'E'])
        params_space = {
                'c1': scipy.stats.expon(scale=0.5),
                'c2': scipy.stats.expon(scale=0.05),
        }
        clf_CV = RandomizedSearchCV(crf, params_space,
                                cv      = cv,
                                verbose = 3,
                                n_jobs  = 3,
                                n_iter  = n_iter,
                                scoring = f1_scorer)
        clf_CV.fit(self.X_train, self.Y_train)
        self.crf = clf_CV.best_estimator_
        return self.crf

    def train(self, sub_train=1.0, c1=0, c2=1, verbose=False):
        crf = self.get_CRF(c1=c1, c2=c2)
        sub_size = int( sub_train*len(self.X_train) )
        if sub_train < 1.0:
            sub_x, sub_y = resample(self.X_train, self.Y_train, n_samples=sub_size, replace=False)
            crf.fit(sub_x, sub_y)
            self.sub_x = sub_x
            self.sub_y = sub_y
            self.crf = crf
            return crf

        crf.fit(self.X_train, self.Y_train)
        self.crf = crf

    def state_features(self, crf_model=None, file_name='state_features.txt'):
        if crf_model == None:
            crf_model = self.crf
        state_features = Counter(crf_model.state_features_).most_common(30)
        f = open(file_name, 'w')
        f.write('Top positive:\n')
        for (attr, label), weight in state_features:
            f.write("%0.6f %-8s %s\n" % (weight, label, attr))

        f.write('Top negative:\n')
        state_features = Counter(crf_model.state_features_).most_common()[-30:]
        for (attr, label), weight in state_features:
            f.write("%0.6f %-8s %s\n" % (weight, label, attr))
        f.close()

    def predict(self, X):
        #  print('pred by learner, not boosting')
        return self.crf.predict(X)

    def predict_prob(self, X):
        return self.crf.predict_marginals(X)

    def predict_file(self, path):
        test_data = Data(path)
        test_data.feature_loader(self.funcs, self.params)
        self.Y_pred = self.clf.predict(test_data.X)
        self.Y_private = test_data.Y
        return self.Y_pred

    def get_score(self, Y_pred=None, Y_private=None, label='E'):
        if ( Y_pred == None ) or ( Y_private == None ):
            Y_pred = self.predict(self.X_private)
            Y_private = self.Y_private
        P = metrics.flat_precision_score(Y_private, Y_pred, pos_label=label)
        R = metrics.flat_recall_score(Y_private, Y_pred, pos_label=label)
        f1 = metrics.flat_f1_score(Y_private, Y_pred, pos_label=label)
        return {'P':P, 'R':R, 'f1':f1}

    def baseline(self):
        self.crf.fit(self.X_train, self.Y_train)
        self.report()

    def report(self, Y_pred=None):
        if Y_pred == None:
            Y_pred = self.predict(self.X_private)
        print(metrics.flat_classification_report(
            self.Y_private, Y_pred, labels=('I', 'E'), digits=4
        ))

class RandomForestLearner(Learner):
    def __init__(self, data, max_dim=100):
        super().__init__(data)
        self.max_dim = max_dim
    def get_CRF(self, c1=0, c2=1):
        clf = RandomForest(n_jobs=8, random_state=self.random_state, max_features=None, n_estimators=3)
        clf.build_index(self.X, max_dim=self.max_dim)
        return clf
    def save_index(self):
        crf = self.get_CRF(c1=1, c2=0)
        crf.save_index(self.X_train, self.Y_train, 'train')
        crf.save_index(self.X_private, self.Y_private, 'test')

class RandomLearner(Learner):
    def get_CRF(self, c1=0, c2=1):
        crf = RandomCRF(
            algorithm                = 'lbfgs',
            max_iterations           = 100,
            all_possible_transitions = True,
            c1 = c1,
            c2 = c2
        )
        return crf

class WeightLearner(Learner):
    def __init__(self, data, random_state=None):
        super().__init__(data, random_state=random_state)
        N = len(self.X_train)
        self.weight_list = [np.longdouble(1/N)]*N
    def split_data(self, train_size=0.7, random_state=None):
        super().split_data(train_size=train_size)
        N = len(self.X_train)
        self.weight_list = [np.longdouble(1/N)]*N
    def get_CRF(self, c1=0, c2=1):
        crf = WeightCRF(
            algorithm                = 'lbfgs',
            max_iterations           = 100,
            all_possible_transitions = True,
            c1 = c1,
            c2 = c2
        )
        return crf
    def train(self, sub_train=1.0, c1=0, c2=1, verbose=False):
        crf = self.get_CRF(c1=c1, c2=c2)
        sub_size = int( sub_train*len(self.X_train) )
        if sub_train < 1.0:
            sub_x, sub_y = resample(self.X_train, self.Y_train, n_samples=sub_size, replace=False)
            crf.fit(sub_x, sub_y, self.weight_list)
            self.sub_x = sub_x
            self.sub_y = sub_y
            self.crf = crf
            return crf
        fit_res = crf.fit(self.X_train, self.Y_train, self.weight_list)
        if fit_res == True:
            self.crf = crf
            return crf
        else:
            logging.info('no model gen')
            return None

class WeightRandonForestLearner(WeightLearner):
    def get_CRF(self, c1=0, c2=1):
        crf = WeightRandomForest(n_jobs=8)
        crf.build_index(self.X)
        return crf
