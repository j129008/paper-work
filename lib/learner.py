from lib.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
from lib.crf import CRF, RandomCRF, WeightCRF, RandomForest
import numpy as np

class Learner(Data):
    def __init__(self, path, train_size=0.6):
        super().__init__(path)
        self.X_train, self.X_private, self.Y_train, self.Y_private = train_test_split(
            self.X, self.Y, test_size=1.0-train_size
        )
    def get_CRF(self, c1=0, c2=1):
        crf = CRF(
            algorithm                = 'lbfgs',
            max_iterations           = 100,
            all_possible_transitions = True,
            c1 = c1,
            c2 = c2
        )
        return crf
    def train_CV(self, c1=0, c2=1, cv=3, n_iter=1, params_space={}):
        crf = self.get_CRF(c1=c1, c2=c2)
        f1_scorer = make_scorer(metrics.flat_f1_score,
                                average='weighted',
                                labels=['I', 'E'])
        clf_CV = RandomizedSearchCV(crf, params_space,
                                cv      = cv,
                                verbose = 3,
                                n_jobs  = 8,
                                n_iter  = n_iter,
                                scoring = f1_scorer)
        clf_CV.fit(self.X_train, self.Y_train)
        clf = clf_CV.best_estimator_
        self.Y_pred = clf.predict(self.X_private)
        return clf

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
        return crf

    def predict(self, X):
        return self.crf.predict(X)

    def predict_prob(self, X):
        return self.crf.predict_prob(X)

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
    def get_CRF(self, c1=0, c2=1):
        clf = RandomForest(n_jobs=8)
        clf.build_index(self.X)
        return clf

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
    def __init__(self, path):
        super().__init__(path)
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
        crf.fit(self.X_train, self.Y_train, self.weight_list)
        self.crf = crf
        return crf
