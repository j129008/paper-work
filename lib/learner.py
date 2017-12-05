from lib.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
from lib.crf import CRF

class Learner(Data):
    def __init__(self, path, train_size=0.6):
        super().__init__(path)
        self.X_train, self.X_private, self.Y_train, self.Y_private = train_test_split(
            self.X, self.Y, test_size=1.0-train_size
        )
    def train_CV(self, c1=0, c2=1, cv=3, n_iter=1, params_space={}):
        crf = CRF(
            algorithm                = 'lbfgs',
            max_iterations           = 100,
            all_possible_transitions = True,
            c1 = c1,
            c2 = c2
        )
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
        self.clf = clf_CV.best_estimator_
        self.Y_pred = self.clf.predict(self.X_private)
        return self.crf

    def train(self, sub_train=1.0, c1=0, c2=1):
        self.crf = CRF(
            algorithm                = 'lbfgs',
            max_iterations           = 100,
            all_possible_transitions = True,
            verbose                  = True,
            c1                       = c1,
            c2                       = c2
        )

        sub_size = int( sub_train*len(self.X_train) )
        if sub_train < 1.0:
            sub_x, sub_y = resample(self.X_train, self.Y_train, n_samples=sub_size)
            self.crf.fit(sub_x, sub_y)
            return self.crf

        self.crf.fit(self.X_train, self.Y_train)
        return self.crf

    def predict_file(self, path):
        test_data = Data(path)
        test_data.feature_loader(self.funcs, self.params)
        self.Y_pred = self.clf.predict(test_data.X)
        self.Y_private = test_data.Y
        return self.Y_pred

    def report(self):
        self.Y_pred = self.crf.predict(self.X_private)
        print(metrics.flat_classification_report(
            self.Y_private, self.Y_pred, labels=('I', 'E'), digits=4
        ))
