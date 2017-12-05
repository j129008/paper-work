from lib.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
from lib.crf import CRF

class Learner(Data):
    def __init__(self, path):
        super().__init__(path)
    def train_CV(self, train_size=0.6, c1=0, c2=1, cv=3, n_iter=1, params_space={}):
        X, X_private, Y, Y_private = train_test_split(
            self.X, self.Y, test_size=1.0-train_size
        )
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
        self.clf = RandomizedSearchCV(crf, params_space,
                                cv      = cv,
                                verbose = 1,
                                n_jobs  = 8,
                                n_iter  = n_iter,
                                scoring = f1_scorer)
        self.clf.fit(X, Y)
        Y_pred = self.clf.best_estimator_.predict(X_private)
        print(metrics.flat_classification_report(
            Y_private, Y_pred, labels=('I', 'E'), digits=4
        ))
